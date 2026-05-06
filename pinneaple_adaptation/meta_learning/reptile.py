from __future__ import annotations
"""Reptile meta-learning trainer for parametric PDE families.

Reptile is a first-order meta-learning algorithm that avoids the second-order
gradients required by full MAML. The outer update simply interpolates the
shared weights toward the task-specific weights after inner training.

Reference
---------
Nichol, Achiam & Schulman (2018). "On First-Order Meta-Learning Algorithms."
https://arxiv.org/abs/1803.02999
"""

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ReptileConfig
from .task_sampler import PDETaskSampler

logger = logging.getLogger(__name__)


class ReptileTrainer:
    """Reptile meta-learning trainer.

    The Reptile outer update is:
        θ ← θ + ε · (θ_task − θ)

    where θ_task is obtained by running ``n_inner_steps`` gradient descent
    steps on a single sampled task using the task-specific physics / data loss.

    Parameters
    ----------
    model : nn.Module
        Base model whose initialisation will be meta-trained.
    config : ReptileConfig
        Hyper-parameters for both inner and outer loops.
    task_sampler : PDETaskSampler
        Provides tasks on demand.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReptileConfig,
        task_sampler: PDETaskSampler,
    ) -> None:
        self.model = model
        self.config = config
        self.task_sampler = task_sampler
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _task_update(self, task: dict) -> Tuple[dict, float]:
        """Train a deep copy of the model on *task* for ``n_inner_steps``.

        Returns the trained parameter dict and the final task loss.
        """
        cfg = self.config
        task_model = copy.deepcopy(self.model).to(cfg.device)
        optimizer = torch.optim.SGD(task_model.parameters(), lr=cfg.inner_lr)

        physics_fn = task.get("physics_fn")
        support = task.get("support", {})

        last_loss = 0.0
        for _ in range(cfg.n_inner_steps):
            optimizer.zero_grad()
            loss = self._compute_task_loss(task_model, physics_fn, support)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        return {k: v.clone().detach() for k, v in task_model.state_dict().items()}, last_loss

    def _compute_task_loss(
        self,
        model: nn.Module,
        physics_fn: Optional[Callable],
        batch: dict,
    ) -> torch.Tensor:
        """Compute physics + optional data loss for one task batch."""
        import numpy as np

        device = self.config.device
        zero = next(model.parameters()).new_zeros(())

        if physics_fn is None:
            return zero

        x_col = batch.get("x_col")
        if x_col is None:
            return zero

        if isinstance(x_col, np.ndarray):
            x_col = torch.from_numpy(x_col).to(device)
        else:
            x_col = x_col.to(device)

        x_col = x_col.requires_grad_(True)
        total, _ = physics_fn(model, {"x_col": x_col})

        # Optional supervised term
        x_data = batch.get("x_data")
        y_data = batch.get("y_data")
        if x_data is not None and y_data is not None:
            if isinstance(x_data, np.ndarray):
                x_data = torch.from_numpy(x_data).to(device)
            if isinstance(y_data, np.ndarray):
                y_data = torch.from_numpy(y_data).to(device)
            out = model(x_data)
            if hasattr(out, "y"):
                out = out.y
            total = total + torch.mean((out - y_data) ** 2)

        return total

    # ------------------------------------------------------------------
    # Outer loop
    # ------------------------------------------------------------------

    def _outer_update(self, task_params: dict) -> None:
        """Reptile outer update: θ ← θ + ε (θ_task − θ)."""
        eps = self.config.epsilon
        current_sd = self.model.state_dict()
        new_sd = {}
        for k in current_sd:
            if k in task_params:
                new_sd[k] = current_sd[k] + eps * (task_params[k].to(current_sd[k].device) - current_sd[k])
            else:
                new_sd[k] = current_sd[k]
        self.model.load_state_dict(new_sd)

    def meta_update(self, tasks: List[dict]) -> dict:
        """Run outer update for a batch of tasks.

        For Reptile, typically ``len(tasks) == 1`` per outer step.
        """
        task_losses = []
        merged_params: Optional[dict] = None

        for task in tasks:
            task_params, loss = self._task_update(task)
            task_losses.append(loss)
            if merged_params is None:
                merged_params = task_params
            else:
                for k in merged_params:
                    merged_params[k] = merged_params[k] + task_params[k]

        if merged_params is not None:
            n = len(tasks)
            if n > 1:
                for k in merged_params:
                    merged_params[k] = merged_params[k] / n
            self._outer_update(merged_params)

        return {"meta_loss": float(sum(task_losses) / max(len(task_losses), 1)), "task_losses": task_losses}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, callbacks: Optional[List] = None) -> dict:
        """Run full Reptile meta-training.

        Returns
        -------
        dict with keys:
            history : list of per-epoch dicts {epoch, meta_loss}
            model   : final meta-trained model
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.model.to(cfg.device)
        self._history = []

        for epoch in range(1, cfg.n_meta_epochs + 1):
            tasks = self.task_sampler.sample_batch(cfg.n_tasks_per_batch)
            info = self.meta_update(tasks)
            record = {"epoch": epoch, **info}
            self._history.append(record)

            if epoch % max(1, cfg.n_meta_epochs // 10) == 0:
                logger.info("Reptile epoch %d/%d  meta_loss=%.4e",
                            epoch, cfg.n_meta_epochs, info["meta_loss"])

            if callbacks:
                for cb in callbacks:
                    cb(epoch, record)

            if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                ckpt_path = f"reptile_meta_epoch{epoch:05d}.pt"
                torch.save({"state_dict": self.model.state_dict(), "epoch": epoch}, ckpt_path)

        return {"history": self._history, "model": self.model}

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def adapt(self, task: dict, n_steps: Optional[int] = None) -> nn.Module:
        """Fast-adapt the meta-model to *task*.

        Returns a deep-copied adapted model; ``self.model`` is unchanged.
        """
        cfg = self.config
        n = n_steps if n_steps is not None else cfg.n_inner_steps
        adapted = copy.deepcopy(self.model).to(cfg.device)
        optimizer = torch.optim.SGD(adapted.parameters(), lr=cfg.inner_lr)

        physics_fn = task.get("physics_fn")
        support = task.get("support", {})

        for _ in range(n):
            optimizer.zero_grad()
            loss = self._compute_task_loss(adapted, physics_fn, support)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        adapted.eval()
        return adapted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save meta-model and config."""
        import json
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {
                "n_inner_steps": self.config.n_inner_steps,
                "inner_lr": self.config.inner_lr,
                "outer_lr": self.config.outer_lr,
                "n_tasks_per_batch": self.config.n_tasks_per_batch,
                "n_meta_epochs": self.config.n_meta_epochs,
                "device": self.config.device,
                "seed": self.config.seed,
                "epsilon": self.config.epsilon,
            },
            "history": self._history,
        }, path)
        logger.info("Saved Reptile meta-model to %s", path)

    @classmethod
    def load(cls, path: str, model_factory: Callable[[], nn.Module], task_sampler=None) -> "ReptileTrainer":
        """Load a saved Reptile trainer.

        Parameters
        ----------
        path : checkpoint path
        model_factory : callable that returns a fresh model instance
        task_sampler : optional PDETaskSampler (required for further training)
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = model_factory()
        model.load_state_dict(ckpt["state_dict"])
        cfg_dict = ckpt.get("config", {})
        cfg = ReptileConfig(**{k: v for k, v in cfg_dict.items() if k in ReptileConfig.__dataclass_fields__})
        trainer = cls(model, cfg, task_sampler or _DummySampler())
        trainer._history = ckpt.get("history", [])
        return trainer


class _DummySampler:
    def sample_batch(self, n):
        return []
