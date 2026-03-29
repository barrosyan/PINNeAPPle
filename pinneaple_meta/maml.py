"""MAML (Model-Agnostic Meta-Learning) trainer for parametric PDE families.

Implements both first-order MAML (FOMAML, default) and full second-order MAML
for physics-informed neural networks.  The inner loop uses
``torch.func.functional_call`` (PyTorch >= 2.0) when available; it falls back
to a manual deep-copy + SGD approach on older versions.

References
----------
* Finn et al. (2017) — "Model-Agnostic Meta-Learning for Fast Adaptation of
  Deep Networks". https://arxiv.org/abs/1703.03400

Example
-------
>>> import torch.nn as nn
>>> from pinneaple_meta.config import MAMLConfig
>>> from pinneaple_meta.task_sampler import PDETaskSampler
>>> from pinneaple_meta.maml import MAMLTrainer
>>>
>>> model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
>>> cfg = MAMLConfig(n_inner_steps=5, first_order=True)
>>> trainer = MAMLTrainer(model, cfg, sampler)
>>> history = trainer.train()
>>> adapted = trainer.adapt(new_task, n_steps=10)
"""
from __future__ import annotations

import copy
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .config import MAMLConfig
from .task_sampler import PDETaskSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect torch.func availability (PyTorch >= 2.0)
# ---------------------------------------------------------------------------
try:
    from torch.func import functional_call, grad  # type: ignore[attr-defined]
    _TORCH_FUNC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_FUNC_AVAILABLE = False


def _params_to_device(
    params: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in params.items()}


def _compute_task_loss(
    model: nn.Module,
    task: dict,
    device: torch.device,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute the physics residual loss on a task's batch.

    If *params* is given and ``torch.func`` is available the model is called
    statelessly via ``functional_call``; otherwise the model's current
    parameters are used directly.

    Parameters
    ----------
    model:
        The base network (used for architecture; weights from *params* if given).
    task:
        Task dict from :class:`PDETaskSampler`.
    device:
        Computation device.
    params:
        Optional parameter dict override.  When *None* the model is called
        normally.
    """
    batch = task["support"]
    x_col = batch["x_col"].to(device)
    physics_fn = task["physics_fn"]

    if params is not None and _TORCH_FUNC_AVAILABLE:
        # Stateless forward via torch.func.functional_call
        def forward(x: torch.Tensor) -> torch.Tensor:
            return functional_call(model, params, (x,))
        loss = physics_fn(forward, x_col)
    else:
        loss = physics_fn(model, x_col)
    return loss


def _compute_query_loss(
    model: nn.Module,
    task: dict,
    device: torch.device,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute physics residual on a task's query set."""
    query = task["query"]
    x_col = query["x_col"].to(device)
    physics_fn = task["physics_fn"]

    if params is not None and _TORCH_FUNC_AVAILABLE:
        def forward(x: torch.Tensor) -> torch.Tensor:
            return functional_call(model, params, (x,))
        loss = physics_fn(forward, x_col)
    else:
        loss = physics_fn(model, x_col)
    return loss


class MAMLTrainer:
    """Trains a model with MAML across a distribution of PDE tasks.

    Parameters
    ----------
    model:
        The neural network to meta-train.  Must be an ``nn.Module``.
    config:
        :class:`~pinneaple_meta.config.MAMLConfig` controlling the algorithm.
    task_sampler:
        :class:`~pinneaple_meta.task_sampler.PDETaskSampler` that produces
        tasks from the target PDE family.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig,
        task_sampler: PDETaskSampler,
    ) -> None:
        self.config = config
        self.task_sampler = task_sampler
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)

        self.model = model.to(self.device)
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), lr=config.outer_lr
        )
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _inner_loop_functional(
        self, task: dict
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Inner loop using ``torch.func.functional_call`` (PyTorch >= 2.0).

        Performs ``n_inner_steps`` gradient steps on the support set using the
        current meta-parameters as initialization, then evaluates the query
        loss on the adapted parameters.

        Returns
        -------
        adapted_params : Dict[str, Tensor]
        query_loss : Tensor
        """
        cfg = self.config
        # Start from a copy of the current meta-parameters
        params: Dict[str, torch.Tensor] = {
            name: p.clone()
            for name, p in self.model.named_parameters()
        }

        for _ in range(cfg.n_inner_steps):
            x_col = task["support"]["x_col"].to(self.device)
            physics_fn = task["physics_fn"]

            def forward(x: torch.Tensor) -> torch.Tensor:
                return functional_call(self.model, params, (x,))

            support_loss = physics_fn(forward, x_col)

            grads = torch.autograd.grad(
                support_loss,
                list(params.values()),
                create_graph=not cfg.first_order,
                allow_unused=True,
            )

            params = {
                name: p - cfg.inner_lr * (g if g is not None else torch.zeros_like(p))
                for (name, p), g in zip(params.items(), grads)
            }

        # Query loss on adapted params
        x_query = task["query"]["x_col"].to(self.device)
        physics_fn = task["physics_fn"]

        def adapted_forward(x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.model, params, (x,))

        query_loss = physics_fn(adapted_forward, x_query)
        return params, query_loss

    def _inner_loop_fallback(
        self, task: dict
    ) -> Tuple[nn.Module, torch.Tensor]:
        """Inner loop via deep-copy + SGD (no ``torch.func``).

        This path is used on PyTorch < 2.0.  It cannot propagate second-order
        gradients through the inner update, so it always behaves as FOMAML.

        Returns
        -------
        adapted_model : nn.Module   (deep copy with adapted weights)
        query_loss : Tensor         (detached from meta-graph — FOMAML only)
        """
        cfg = self.config
        adapted = copy.deepcopy(self.model)
        inner_opt = optim.SGD(adapted.parameters(), lr=cfg.inner_lr)

        for _ in range(cfg.n_inner_steps):
            inner_opt.zero_grad()
            x_col = task["support"]["x_col"].to(self.device)
            loss = task["physics_fn"](adapted, x_col)
            loss.backward()
            inner_opt.step()

        adapted.eval()
        x_query = task["query"]["x_col"].to(self.device)
        query_loss = task["physics_fn"](adapted, x_query)
        return adapted, query_loss

    def _inner_loop(
        self, task: dict
    ) -> Tuple[object, torch.Tensor]:
        """Dispatch to functional or fallback inner loop."""
        if _TORCH_FUNC_AVAILABLE:
            return self._inner_loop_functional(task)
        else:
            return self._inner_loop_fallback(task)

    # ------------------------------------------------------------------
    # Outer loop / meta-update
    # ------------------------------------------------------------------

    def meta_update(self, tasks: List[dict]) -> dict:
        """Perform one meta-update step over *tasks*.

        Parameters
        ----------
        tasks:
            List of task dicts from :class:`PDETaskSampler`.

        Returns
        -------
        dict
            ``{"meta_loss": float, "task_losses": List[float]}``
        """
        self.meta_optimizer.zero_grad()
        task_losses: List[float] = []
        meta_loss = torch.tensor(0.0, device=self.device)

        for task in tasks:
            _, query_loss = self._inner_loop(task)
            meta_loss = meta_loss + query_loss
            task_losses.append(float(query_loss.detach()))

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()

        return {
            "meta_loss": float(meta_loss.detach()),
            "task_losses": task_losses,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, callbacks: Optional[List[Callable]] = None) -> dict:
        """Run the full MAML meta-training loop.

        Parameters
        ----------
        callbacks:
            Optional list of callables invoked at the end of each epoch as
            ``callback(epoch, metrics_dict, trainer)``.

        Returns
        -------
        dict
            ``{"history": List[dict]}`` where each element contains epoch
            metrics.
        """
        cfg = self.config
        callbacks = callbacks or []
        logger.info(
            "Starting MAML meta-training: %d epochs, %d tasks/batch, "
            "%d inner steps, first_order=%s",
            cfg.n_meta_epochs,
            cfg.n_tasks_per_batch,
            cfg.n_inner_steps,
            cfg.first_order,
        )

        for epoch in range(1, cfg.n_meta_epochs + 1):
            t0 = time.perf_counter()
            tasks = self.task_sampler.sample_batch(cfg.n_tasks_per_batch)
            metrics = self.meta_update(tasks)
            metrics["epoch"] = epoch
            metrics["elapsed"] = time.perf_counter() - t0
            self.history.append(metrics)

            if epoch % max(cfg.n_meta_epochs // 10, 1) == 0:
                logger.info(
                    "Epoch %d/%d — meta_loss=%.6f",
                    epoch, cfg.n_meta_epochs, metrics["meta_loss"],
                )

            if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                ckpt_path = f"maml_checkpoint_epoch{epoch:05d}.pt"
                self.save(ckpt_path)
                logger.info("Checkpoint saved → %s", ckpt_path)

            for cb in callbacks:
                cb(epoch, metrics, self)

        return {"history": self.history}

    # ------------------------------------------------------------------
    # Fast adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        task: dict,
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> nn.Module:
        """Fast-adapt the meta-model to *task*.

        A deep copy of the meta-model is fine-tuned on *task*'s support set
        for *n_steps* gradient steps and returned.  The original model is
        unchanged.

        Parameters
        ----------
        task:
            Task dict containing at least ``"support"`` and ``"physics_fn"``.
        n_steps:
            Number of inner-loop adaptation steps.  Defaults to
            ``config.n_inner_steps``.
        lr:
            Inner learning rate override.  Defaults to ``config.inner_lr``.

        Returns
        -------
        nn.Module
            Adapted copy of the meta-model (in ``eval`` mode).
        """
        cfg = self.config
        n_steps = n_steps if n_steps is not None else cfg.n_inner_steps
        lr = lr if lr is not None else cfg.inner_lr

        adapted = copy.deepcopy(self.model).to(self.device)
        inner_opt = optim.SGD(adapted.parameters(), lr=lr)

        adapted.train()
        for _ in range(n_steps):
            inner_opt.zero_grad()
            x_col = task["support"]["x_col"].to(self.device)
            loss = task["physics_fn"](adapted, x_col)
            loss.backward()
            inner_opt.step()

        adapted.eval()
        return adapted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the meta-model weights and config to *path*.

        Parameters
        ----------
        path:
            File path (``*.pt`` or ``*.pth`` recommended).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.meta_optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        logger.debug("MAMLTrainer saved to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        model_factory: Callable[[], nn.Module],
        task_sampler: Optional[PDETaskSampler] = None,
    ) -> "MAMLTrainer":
        """Load a previously saved :class:`MAMLTrainer`.

        Parameters
        ----------
        path:
            Path to the ``*.pt`` checkpoint created by :meth:`save`.
        model_factory:
            Zero-argument callable that returns an *untrained* model with the
            same architecture as the saved one (used to instantiate the module
            before loading state dict).
        task_sampler:
            Optional task sampler to attach to the restored trainer.  Can be
            ``None`` if you only intend to call :meth:`adapt` or
            :meth:`predict`.

        Returns
        -------
        MAMLTrainer
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config: MAMLConfig = ckpt["config"]
        model = model_factory()
        model.load_state_dict(ckpt["model_state"])

        dummy_sampler = task_sampler or PDETaskSampler(
            param_ranges={"__dummy__": (0.0, 1.0)},
            physics_fn_factory=lambda p: (lambda m, x: torch.tensor(0.0)),
        )
        trainer = cls(model, config, dummy_sampler)
        trainer.meta_optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.history = ckpt.get("history", [])
        logger.info("MAMLTrainer loaded from %s", path)
        return trainer
