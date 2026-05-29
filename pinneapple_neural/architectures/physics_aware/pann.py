from __future__ import annotations
"""Physics-aware neural network with PDE-informed losses."""
from typing import Dict, Optional, Callable, Any, Tuple, Mapping, Union

import torch
import torch.nn as nn

from .base import PhysicsAwareBase, PhysicsAwareOutput


PhysicsLossReturn = Union[
    torch.Tensor,  # scalar tensor
    Tuple[torch.Tensor, Mapping[str, float]],  # (total_loss, components)
]


class PhysicsAwareNeuralNetwork(PhysicsAwareBase):
    """
    Physics-Aware Neural Network:

    Supports two physics loss contracts:

    (A) PINNFactory-style:
        physics_loss_fn(model, batch) -> (total_loss, components_dict)

    (B) Legacy/MVP-style:
        physics_loss_fn(pred, batch) -> scalar_loss

    Notes:
      - If using PINNFactory, pass backbone=model (e.g., PINN) and set y_true=None
        or avoid duplicating data loss outside.
    """
    def __init__(
        self,
        backbone: nn.Module,
        *,
        physics_loss_fn: Optional[Callable[..., PhysicsLossReturn]] = None,
        physics_weight: float = 1.0,
        physics_components_prefix: str = "physics/",
    ):
        super().__init__()
        self.backbone = backbone
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = float(physics_weight)
        self.physics_components_prefix = str(physics_components_prefix)

    @staticmethod
    def _zero_scalar_like(ref: torch.Tensor) -> torch.Tensor:
        return ref.sum() * 0.0  # dtype/device-safe scalar

    @staticmethod
    def _to_scalar_tensor(v: Any, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(v):
            t = v.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
        if t.ndim != 0:
            t = t.mean()
        return t

    def _call_physics_loss(
        self,
        *,
        y_pred: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Try PINNFactory-style first: physics_loss_fn(model, batch) -> (loss, components)
        Fallback to legacy: physics_loss_fn(pred, batch) -> loss
        """
        assert self.physics_loss_fn is not None

        # 1) Factory-style: (model, batch) -> (loss, components)
        try:
            out = self.physics_loss_fn(self.backbone, batch)
            if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]):
                loss_t, comps = out
                comps_dict = dict(comps) if comps is not None else {}
                return loss_t, {k: float(v) for k, v in comps_dict.items()}
            if torch.is_tensor(out):
                # Some users may return only loss even with (model, batch) signature
                return out, {}
        except TypeError:
            # likely wrong signature, fallback below
            pass

        # 2) Legacy-style: (pred, batch) -> loss
        out2 = self.physics_loss_fn(y_pred, batch)
        if isinstance(out2, tuple) and len(out2) == 2 and torch.is_tensor(out2[0]):
            loss_t, comps = out2
            comps_dict = dict(comps) if comps is not None else {}
            return loss_t, {k: float(v) for k, v in comps_dict.items()}
        if not torch.is_tensor(out2):
            raise TypeError("physics_loss_fn must return a torch.Tensor loss (optionally with components dict).")
        return out2, {}

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
        return_loss: bool = False,
    ) -> PhysicsAwareOutput:
        y = self.backbone(x)

        extras: Dict[str, Any] = {}
        losses: Dict[str, torch.Tensor] = {}

        if return_loss:
            total = self._zero_scalar_like(y)

            if y_true is not None:
                data = self.mse(y, y_true)
                losses["data"] = data
                total = total + data

            if self.physics_loss_fn is not None and batch is not None:
                pl, comps = self._call_physics_loss(y_pred=y, batch=batch)
                pl = self._to_scalar_tensor(pl, total)

                losses["physics"] = pl
                total = total + self.physics_weight * pl

                # keep components for logging without changing factory
                if comps:
                    extras["physics_components"] = comps
                    # optional: also mirror as scalar tensors in losses with prefix
                    for k, v in comps.items():
                        kk = k if str(k).startswith(self.physics_components_prefix) else f"{self.physics_components_prefix}{k}"
                        losses[kk] = self._to_scalar_tensor(v, total)

            losses["total"] = total

        return PhysicsAwareOutput(y=y, losses=losses, extras=extras)