"""Loss functions for supervised and physics-aware training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _unwrap_pred(y_hat: Any) -> torch.Tensor:
    """
    Accepts either:
      - torch.Tensor
      - objects with common prediction attributes: y, pred, out, recon, x_hat
    Returns a torch.Tensor prediction.
    """
    if isinstance(y_hat, torch.Tensor):
        return y_hat

    for attr in ("y", "pred", "out", "recon", "x_hat", "y_hat"):
        if hasattr(y_hat, attr):
            v = getattr(y_hat, attr)
            if isinstance(v, torch.Tensor):
                return v

    raise TypeError(
        f"Expected y_hat to be a Tensor or have a Tensor attribute among "
        f"[y, pred, out, recon, x_hat, y_hat]. Got: {type(y_hat).__name__}"
    )


@dataclass
class SupervisedLoss:
    kind: str = "mse"  # "mse" | "mae"

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "mae":
            return torch.mean(torch.abs(y_hat - y))
        return torch.mean((y_hat - y) ** 2)


@dataclass
class PhysicsLossHook:
    physics_loss_fn: Any

    def __call__(self, model: nn.Module, y_hat: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # keep y_hat as-is; some physics hooks may want ModelOutput
        fn = self.physics_loss_fn
        out: Dict[str, torch.Tensor] = {}
        try:
            loss, comps = fn(model, batch)
            out["physics"] = loss
            for k, v in (comps or {}).items():
                if isinstance(v, (float, int)):
                    out[f"physics_{k}"] = torch.tensor(float(v), device=loss.device)
        except TypeError:
            pred = _unwrap_pred(y_hat)
            loss = fn(pred, batch)
            out["physics"] = loss
        return out


@dataclass
class CombinedLoss:
    supervised: SupervisedLoss = field(default_factory=lambda: SupervisedLoss("mse"))  # <-- Python 3.13 safe
    physics: Optional[PhysicsLossHook] = None
    w_supervised: float = 1.0
    w_physics: float = 1.0

    def __call__(self, model: nn.Module, y_hat: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pred = _unwrap_pred(y_hat)
        y = batch.get("y")

        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if y is not None:
            l_sup = self.supervised(pred, y)
            losses["supervised"] = l_sup
            total = total + self.w_supervised * l_sup

        if self.physics is not None:
            phy = self.physics(model, y_hat, batch)
            if "physics" in phy:
                losses.update(phy)
                total = total + self.w_physics * phy["physics"]

        losses["total"] = total
        return losses
