from __future__ import annotations
"""MetaModel — high-level wrapper for a trained meta-model.

Provides a convenient interface for adaptation and prediction regardless of
whether the underlying trainer is MAML or Reptile.
"""

import copy
import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MetaModel:
    """Wraps a trained meta-model for easy adaptation and inference.

    Parameters
    ----------
    model : nn.Module
        Meta-trained model (initialisation optimised for fast adaptation).
    meta_type : str
        "reptile" or "maml" — used for labelling / serialisation.
    device : str
        Inference device.

    Example
    -------
    >>> meta = MetaModel(trained_model, meta_type="reptile")
    >>> adapted = meta.adapt(physics_fn, data, n_steps=20)
    >>> y_pred = meta.predict(x_test, adapted_model=adapted)
    """

    def __init__(
        self,
        model: nn.Module,
        meta_type: str = "reptile",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.meta_type = meta_type
        self.device = device

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        physics_fn: Optional[Callable] = None,
        data: Optional[dict] = None,
        n_steps: int = 10,
        lr: float = 0.01,
        task: Optional[dict] = None,
    ) -> nn.Module:
        """Adapt the meta-model to a new task.

        Parameters
        ----------
        physics_fn : callable(model, batch) -> (loss, components)
        data : dict with "x" (coords) and optionally "y" (targets)
        n_steps : number of gradient steps
        lr : learning rate for adaptation
        task : alternative — pass a full task dict (overrides physics_fn/data)

        Returns
        -------
        Adapted nn.Module (deep copy; original is unchanged).
        """
        adapted = copy.deepcopy(self.model).to(self.device)
        optimizer = torch.optim.Adam(adapted.parameters(), lr=lr)

        if task is not None:
            physics_fn = task.get("physics_fn", physics_fn)
            data = task.get("support", data)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = self._task_loss(adapted, physics_fn, data or {})
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

        adapted.eval()
        return adapted

    def _task_loss(
        self,
        model: nn.Module,
        physics_fn: Optional[Callable],
        batch: dict,
    ) -> torch.Tensor:
        zero = next(model.parameters()).new_zeros(())

        x = batch.get("x") or batch.get("x_col")
        if x is None:
            return zero

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        x = x.to(self.device).requires_grad_(True)

        total = zero

        if physics_fn is not None:
            try:
                res = physics_fn(model, {"x_col": x})
                if isinstance(res, tuple):
                    total = total + res[0]
                elif torch.is_tensor(res):
                    total = total + res
            except Exception as e:
                logger.debug("physics_fn failed during adapt: %s", e)

        y = batch.get("y")
        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y.astype(np.float32)).to(self.device)
            out = model(x.detach())
            if hasattr(out, "y"):
                out = out.y
            total = total + torch.mean((out - y) ** 2)

        return total

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x: np.ndarray,
        adapted_model: Optional[nn.Module] = None,
    ) -> np.ndarray:
        """Run inference.

        Parameters
        ----------
        x : (N, D) numpy array of input coordinates
        adapted_model : if provided, use this adapted copy; else use base model

        Returns
        -------
        (N, F) numpy array of predictions
        """
        m = (adapted_model or self.model).to(self.device)
        m.eval()
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        with torch.no_grad():
            out = m(x_t)
            if hasattr(out, "y"):
                out = out.y
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, dtype=torch.float32)
        return out.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save meta-model weights and metadata."""
        torch.save({
            "state_dict": self.model.state_dict(),
            "meta_type": self.meta_type,
            "device": self.device,
        }, path)
        logger.info("MetaModel saved to %s", path)

    @classmethod
    def load(cls, path: str, model_factory: Callable[[], nn.Module]) -> "MetaModel":
        """Load a saved MetaModel.

        Parameters
        ----------
        path : checkpoint file path
        model_factory : callable returning a fresh model instance
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = model_factory()
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model=model,
            meta_type=ckpt.get("meta_type", "reptile"),
            device=ckpt.get("device", "cpu"),
        )
