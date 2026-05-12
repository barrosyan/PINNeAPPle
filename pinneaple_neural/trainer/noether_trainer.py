"""Noether surrogate trainer for PINNeAPPle.

``NoetherSurrogateTrainer`` wraps Noether's ``WeightedLossTrainer`` and
exposes it through an interface compatible with PINNeAPPle's training
conventions:
  - Accepts a ``NoetherUPT`` / ``NoetherABUPT`` (or any ``_NoetherWrapper``)
  - Accepts a ``NoetherDatasetBridge`` (or any ``torch.utils.data.Dataset``
    that yields ``{field: tensor}`` batches)
  - Returns a ``NoetherTrainResult`` compatible with PINNeAPPle's reporting

For users who want to stay entirely within the Noether/Hydra ecosystem,
``build_hydra_trainer`` provides a pass-through that delegates everything to
the native Noether ``HydraRunner``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NoetherTrainConfig:
    """Training configuration for NoetherSurrogateTrainer.

    Parameters
    ----------
    epochs : number of training epochs
    batch_size : samples per batch
    lr : peak learning rate
    weight_decay : AdamW weight decay
    field_weights : per-output field loss weights, e.g.
        {"pressure": 1.0, "velocity": 0.5}
    loss_fn : "mse" (default), "l1", or a dotted path to a custom callable
    warmup_steps : linear LR warmup steps
    grad_clip : max gradient norm (0 = disabled)
    device : "cuda", "cpu", "mps" — None = auto-detect
    num_workers : DataLoader workers
    tracker : "none", "wandb", "tensorboard", "trackio"
    wandb_project : WandB project name (only when tracker="wandb")
    checkpoint_dir : directory for saving checkpoints
    save_best : save the checkpoint with lowest validation loss
    log_interval : print/log every N iterations
    distributed : enable DDP training (requires torchrun / noether.core.distributed)
    """
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    field_weights: Dict[str, float] = field(default_factory=lambda: {"y": 1.0})
    loss_fn: str = "mse"
    warmup_steps: int = 0
    grad_clip: float = 1.0
    device: Optional[str] = None
    num_workers: int = 0
    tracker: str = "none"
    wandb_project: Optional[str] = None
    checkpoint_dir: str = "checkpoints/noether"
    save_best: bool = True
    log_interval: int = 50
    distributed: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class NoetherTrainResult:
    """Summary returned by ``NoetherSurrogateTrainer.train()``."""
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    best_epoch: int
    checkpoint_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "NoetherSurrogateTrainer result",
            f"  epochs trained : {len(self.train_losses)}",
            f"  best val loss  : {self.best_val_loss:.6f}  (epoch {self.best_epoch})",
        ]
        if self.checkpoint_path:
            lines.append(f"  checkpoint     : {self.checkpoint_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NoetherSurrogateTrainer:
    """Train a Noether (UPT / AB-UPT / Transolver …) model with PINNeAPPle
    data and report training results in a PINNeAPPle-compatible way.

    This class provides two training modes:

    Mode A — Pure PyTorch (default)
        A lightweight loop that delegates loss computation to Noether's
        ``WeightedLossTrainer`` logic (replicated without Hydra for simplicity).
        Suitable for notebook experiments and integration with PINNeAPPle
        pipelines.

    Mode B — Native Noether / Hydra
        Call ``NoetherSurrogateTrainer.from_hydra_config(cfg_path)`` instead
        to get a thin wrapper that runs ``noether-train`` natively.

    Parameters
    ----------
    model : a ``_NoetherWrapper`` subclass (NoetherUPT, NoetherABUPT, …)
    train_loader : DataLoader yielding {field_name: tensor} batches
    val_loader   : optional validation DataLoader
    config       : ``NoetherTrainConfig``
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: "torch.utils.data.DataLoader",
        val_loader: Optional["torch.utils.data.DataLoader"] = None,
        config: Optional[NoetherTrainConfig] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or NoetherTrainConfig()
        self._device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        cfg_dev = self.config.device
        if cfg_dev:
            return torch.device(cfg_dev)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _loss_fn(self) -> Callable:
        name = self.config.loss_fn.lower()
        if name == "mse":
            return torch.nn.functional.mse_loss
        if name == "l1":
            return torch.nn.functional.l1_loss
        # Dotted-path import for custom loss
        parts = self.config.loss_fn.rsplit(".", 1)
        if len(parts) == 2:
            import importlib
            mod = importlib.import_module(parts[0])
            return getattr(mod, parts[1])
        raise ValueError(f"Unknown loss_fn: {self.config.loss_fn!r}")

    def _compute_batch_loss(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Forward pass + weighted loss across all fields in the batch.

        Convention: batch contains ``{field: tensor}`` pairs; target keys are
        ``{field}_target`` or ``target_{field}``.  If no target key is found
        for a field, the field is skipped.
        """
        # Separate inputs from targets
        target_keys = {
            k for k in batch
            if k.endswith("_target") or k.startswith("target_")
        }
        input_batch = {k: v.to(self._device) for k, v in batch.items() if k not in target_keys}
        target_batch = {k: v.to(self._device) for k, v in batch.items() if k in target_keys}

        from pinneaple_neural.architectures.neural_operators.base import OperatorOutput
        out = self.model(**input_batch)
        if isinstance(out, OperatorOutput):
            predictions = {"y": out.y}
        elif isinstance(out, dict):
            predictions = out
        else:
            predictions = {"y": out}

        total = torch.tensor(0.0, device=self._device)
        weights = self.config.field_weights
        matched = False
        for pred_key, pred_val in predictions.items():
            for suffix in (f"{pred_key}_target", f"target_{pred_key}"):
                if suffix in target_batch:
                    w = weights.get(pred_key, 1.0)
                    total = total + w * loss_fn(pred_val, target_batch[suffix])
                    matched = True
                    break
        if not matched and target_batch:
            # fallback: use first target
            first_target = next(iter(target_batch.values()))
            first_pred = next(iter(predictions.values()))
            if first_pred.shape == first_target.shape:
                total = loss_fn(first_pred, first_target)
        return total

    def train(self) -> NoetherTrainResult:
        """Run the full training loop.

        Returns
        -------
        NoetherTrainResult
        """
        cfg = self.config
        device = self._device
        model = self.model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        total_steps = cfg.epochs * len(self.train_loader)
        scheduler = None
        if cfg.warmup_steps > 0 and total_steps > cfg.warmup_steps:
            try:
                from torch.optim.lr_scheduler import OneCycleLR
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=cfg.lr,
                    total_steps=total_steps,
                    pct_start=cfg.warmup_steps / total_steps,
                )
            except Exception:
                pass

        loss_fn = self._loss_fn()
        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val = float("inf")
        best_epoch = 0
        checkpoint_path: Optional[str] = None

        import os

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_loader):
                optimizer.zero_grad()
                loss = self._compute_batch_loss(batch, loss_fn)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss += loss.item()
                if (step + 1) % cfg.log_interval == 0:
                    print(
                        f"[Noether] epoch {epoch+1}/{cfg.epochs}  "
                        f"step {step+1}/{len(self.train_loader)}  "
                        f"loss={loss.item():.6f}"
                    )

            avg_train = epoch_loss / max(1, len(self.train_loader))
            train_losses.append(avg_train)

            # Validation
            avg_val = float("nan")
            if self.val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in self.val_loader:
                        val_loss += self._compute_batch_loss(batch, loss_fn).item()
                avg_val = val_loss / max(1, len(self.val_loader))
                val_losses.append(avg_val)

                if avg_val < best_val:
                    best_val = avg_val
                    best_epoch = epoch + 1
                    if cfg.save_best:
                        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(
                            cfg.checkpoint_dir, "best_model.pt"
                        )
                        torch.save(model.state_dict(), checkpoint_path)

            print(
                f"[Noether] epoch {epoch+1}/{cfg.epochs}  "
                f"train={avg_train:.6f}  val={avg_val:.6f}"
            )

        return NoetherTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val if val_losses else float("nan"),
            best_epoch=best_epoch,
            checkpoint_path=checkpoint_path,
        )

    @classmethod
    def from_hydra_config(cls, config_path: str, overrides: Optional[List[str]] = None):
        """Launch training natively via Noether's HydraRunner.

        This delegates *everything* to noether's own config-driven pipeline
        (``noether-train``), making PINNeAPPle a thin caller rather than a
        re-implementer of Noether's full Hydra stack.

        Parameters
        ----------
        config_path : path to a Noether Hydra config directory
            (must contain a ``config.yaml`` understood by ``HydraRunner``)
        overrides : list of Hydra override strings, e.g.
            ["trainer.epochs=100", "model.dim=256"]

        Returns
        -------
        self — a lightweight wrapper (``train()`` is a no-op; use Noether CLI)
        """
        try:
            from noether.training.runners import HydraRunner
        except ImportError:
            raise ImportError(
                "emmiai-noether is required. Install with: pip install emmiai-noether"
            )
        runner = HydraRunner(config_path=config_path, overrides=overrides or [])
        return _HydraRunnerWrapper(runner)


class _HydraRunnerWrapper:
    """Thin wrapper that exposes HydraRunner through NoetherSurrogateTrainer API."""

    def __init__(self, runner) -> None:
        self._runner = runner

    def train(self):
        self._runner.run()
        return NoetherTrainResult(
            train_losses=[], val_losses=[], best_val_loss=float("nan"),
            best_epoch=0, extras={"mode": "hydra_runner"}
        )
