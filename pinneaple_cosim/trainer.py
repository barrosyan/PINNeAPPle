"""CoSimTrainer — trains co-simulation graphs using PINNeAPPle's training stack.

Integrations:
  - ``TrainConfig``         hyperparameters (lr, epochs, grad_clip, device, seed, …)
  - ``WeightScheduler``     adaptive loss balancing (self-adaptive, GradNorm, NTK, …)
  - ``EarlyStopping``       patience-based early exit on val loss
  - ``ModelCheckpoint``     saves the best model weights across all nodes
  - ``CoSimLoss``           data + physics + coupling composite loss

Training loop (BPTT):
  For each epoch, the trainer unrolls ``n_unroll`` co-simulation steps while
  retaining the PyTorch computational graph.  The composite loss is accumulated
  over the rollout, then a single backward pass updates all PINN/surrogate
  parameters in the graph.

Usage::

    from pinneaple_cosim import CoSimGraph, CoSimEngine, CoSimLoss
    from pinneaple_cosim.trainer import CoSimTrainer
    from pinneaple_train.trainer import TrainConfig

    trainer = CoSimTrainer(
        graph=graph,
        engine=engine,
        criterion=CoSimLoss(data_weight=1.0, physics_weight=2.0),
    )

    result = trainer.fit(
        cfg=TrainConfig(epochs=500, lr=1e-3, grad_clip=1.0),
        n_unroll=10,
        dt=0.02,
        initial_ports={"mass": {"x": torch.zeros(1,1), "v": torch.zeros(1,1)}},
        targets_fn=lambda t, pv: {"mass.x": x_ref(t)},          # optional data
        val_targets_fn=lambda t, pv: {"mass.x": x_ref_val(t)},  # optional val
    )
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .engine import CoSimEngine
from .graph import CoSimGraph
from .losses import CoSimLoss
from .recorder import TrajectoryRecorder


class CoSimTrainer:
    """Trains a differentiable co-simulation graph with PINNeAPPle's training stack.

    Args:
        graph:       the ``CoSimGraph`` whose node parameters will be optimised.
        engine:      the ``CoSimEngine`` used for rollouts.
        criterion:   composite loss; defaults to
                     ``CoSimLoss(data=1.0, physics=1.0, coupling=0.1)``.
        weight_scheduler_cfg: optional ``WeightSchedulerConfig`` for adaptive
                     loss balancing.  When *None*, fixed weights are used.
        early_stopping: optional ``EarlyStopping`` callback.
        checkpoint:  optional ``ModelCheckpoint`` callback.
        verbose:     print epoch summary every ``log_every`` epochs.
        log_every:   logging interval (default 50 epochs).
    """

    def __init__(
        self,
        graph: CoSimGraph,
        engine: CoSimEngine,
        criterion: Optional[CoSimLoss] = None,
        weight_scheduler_cfg: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        checkpoint: Optional[Any] = None,
        verbose: bool = True,
        log_every: int = 50,
    ) -> None:
        self.graph = graph
        self.engine = engine
        self.criterion = criterion or CoSimLoss(
            data_weight=1.0, physics_weight=1.0, coupling_weight=0.1
        )
        self._ws_cfg = weight_scheduler_cfg
        self._early_stopping = early_stopping
        self._checkpoint = checkpoint
        self.verbose = verbose
        self.log_every = log_every

        self._history: Dict[str, List[float]] = {
            "train_total": [], "train_physics": [],
            "train_data": [],  "val_total": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        cfg: Any,
        *,
        n_unroll: int = 10,
        dt: float = 0.01,
        t0: float = 0.0,
        initial_ports: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        targets_fn: Optional[Callable[[float, Dict], Dict[str, torch.Tensor]]] = None,
        val_targets_fn: Optional[Callable[[float, Dict], Dict[str, torch.Tensor]]] = None,
        n_val_steps: int = 20,
    ) -> Dict[str, Any]:
        """Run the training loop.

        Args:
            cfg:          ``TrainConfig`` (or any object with ``epochs``, ``lr``,
                          ``weight_decay``, ``grad_clip``, ``device``, ``seed``).
            n_unroll:     number of co-sim steps per BPTT rollout.
            dt:           time step size (seconds or non-dimensional).
            t0:           rollout start time.
            initial_ports: ``{node: {port: tensor}}`` initial state.
            targets_fn:   ``(t, port_values) -> {"node.port": tensor}`` ground-truth
                          at each step.  Pass *None* for physics-only training.
            val_targets_fn: same as *targets_fn* but used only for validation.
            n_val_steps:  number of steps for the validation rollout.

        Returns:
            Dict with keys ``"best_val"``, ``"best_path"``, ``"history"``.
        """
        device = getattr(cfg, "device", "cpu")
        seed   = getattr(cfg, "seed", None)
        if seed is not None:
            torch.manual_seed(seed)

        # Build optimizer
        params = list(self.graph.trainable_parameters())
        if not params:
            raise RuntimeError(
                "CoSimGraph has no trainable parameters. "
                "Add at least one TorchNode, PINNNode, or adapter node."
            )
        optimizer = torch.optim.Adam(
            params,
            lr=getattr(cfg, "lr", 1e-3),
            weight_decay=getattr(cfg, "weight_decay", 0.0),
        )

        # Optional weight scheduler
        weight_scheduler = self._build_weight_scheduler(params)

        epochs    = getattr(cfg, "epochs", 100)
        grad_clip = getattr(cfg, "grad_clip", 0.0)
        best_val  = float("inf")
        best_path: Optional[str] = None

        for epoch in range(epochs):
            # ---- Training rollout ----------------------------------------
            train_loss, train_info = self._rollout(
                optimizer, grad_clip, n_unroll, dt, t0,
                initial_ports, targets_fn, weight_scheduler, training=True,
            )

            # ---- Validation rollout ---------------------------------------
            if val_targets_fn is not None or (epoch % self.log_every == 0):
                val_loss, _ = self._rollout(
                    None, 0.0, n_val_steps, dt, t0,
                    initial_ports, val_targets_fn, None, training=False,
                )
            else:
                val_loss = float("nan")

            self._history["train_total"].append(train_loss)
            self._history["train_physics"].append(train_info.get("physics", 0.0))
            self._history["train_data"].append(train_info.get("data", 0.0))
            self._history["val_total"].append(val_loss)

            # ---- Checkpoint -----------------------------------------------
            if val_loss < best_val:
                best_val = val_loss
                if self._checkpoint is not None:
                    best_path = self._save_checkpoint(epoch, val_loss)

            # ---- Early stopping -------------------------------------------
            if self._early_stopping is not None:
                if self._check_early_stop(epoch, val_loss):
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch+1}.")
                    break

            # ---- Logging --------------------------------------------------
            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(
                    f"  Epoch {epoch+1:5d}/{epochs} | "
                    f"train={train_loss:.5f}  "
                    f"physics={train_info.get('physics', 0.0):.5f}  "
                    f"data={train_info.get('data', 0.0):.5f}  "
                    f"val={val_loss:.5f}"
                )

        return {
            "best_val":  best_val,
            "best_path": best_path,
            "history":   self._history,
        }

    @property
    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout(
        self,
        optimizer: Optional[torch.optim.Optimizer],
        grad_clip: float,
        n_steps: int,
        dt: float,
        t0: float,
        initial_ports: Optional[Dict],
        targets_fn: Optional[Callable],
        weight_scheduler: Optional[Any],
        training: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Run a rollout of n_steps, accumulate loss, optionally back-prop."""
        self.engine.reset()
        if initial_ports:
            self.engine.initialize_ports(initial_ports)

        total_loss = torch.tensor(0.0)
        total_info: Dict[str, float] = {}

        # Physics loss always requires autograd (Laplacian, gradients).
        # We use enable_grad() throughout and detach results when not training.
        with torch.enable_grad():
            for step_i in range(n_steps):
                t = t0 + step_i * dt
                port_vals = self.engine.step(t, dt)

                targets = targets_fn(t + dt, port_vals) if targets_fn else {}
                loss, info = self.criterion(port_vals, self.graph, targets)

                for k, v in info.items():
                    total_info[k] = total_info.get(k, 0.0) + v

                if training:
                    total_loss = total_loss + loss

        if training and optimizer is not None:
            mean_loss = total_loss / n_steps

            # Apply weight scheduler if present
            if weight_scheduler is not None:
                raw = {
                    "data":     torch.tensor(total_info.get("data", 0.0) / n_steps),
                    "physics":  torch.tensor(total_info.get("physics", 0.0) / n_steps),
                    "coupling": torch.tensor(total_info.get("coupling", 0.0) / n_steps),
                }
                mean_loss = weight_scheduler.step(raw)

            optimizer.zero_grad()
            mean_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.graph.trainable_parameters()), grad_clip
                )
            optimizer.step()
            scalar = float(mean_loss.detach())
        else:
            scalar = sum(total_info.values()) / max(n_steps, 1)

        avg_info = {k: v / n_steps for k, v in total_info.items()}
        return scalar, avg_info

    def _build_weight_scheduler(self, params: List) -> Optional[Any]:
        if self._ws_cfg is None:
            return None
        try:
            from pinneaple_train.weight_scheduler import WeightScheduler
            import torch.nn as nn

            class _FakeModel(nn.Module):
                def forward(self, x):
                    return x

            return WeightScheduler(
                model=_FakeModel(),
                loss_names=["data", "physics", "coupling"],
                config=self._ws_cfg,
            )
        except Exception:
            return None

    def _save_checkpoint(self, epoch: int, val_loss: float) -> Optional[str]:
        try:
            path = self._checkpoint.path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                "epoch": epoch,
                "val_loss": val_loss,
                "nodes": {
                    name: {
                        "model_state": (
                            node.model.state_dict()
                            if hasattr(node, "model") else {}
                        )
                    }
                    for name, node in self.graph.nodes.items()
                },
            }
            torch.save(state, path)
            return path
        except Exception:
            return None

    def _check_early_stop(self, epoch: int, val_loss: float) -> bool:
        try:
            logs = {"val_total": val_loss}
            self._early_stopping(logs, epoch)
            return getattr(self._early_stopping, "stopped", False)
        except Exception:
            return False

    def load_best_checkpoint(self, path: str) -> None:
        """Restore all node model weights from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        for name, state in ckpt.get("nodes", {}).items():
            if name in self.graph.nodes:
                node = self.graph.node(name)
                if hasattr(node, "model") and state.get("model_state"):
                    node.model.load_state_dict(state["model_state"])
