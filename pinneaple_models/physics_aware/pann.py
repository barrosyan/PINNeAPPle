from __future__ import annotations
"""Physics-aware neural network with PDE-informed losses."""
from typing import Dict, Optional, Callable, Any

import torch
import torch.nn as nn

from .base import PhysicsAwareBase, PhysicsAwareOutput


class PhysicsAwareNeuralNetwork(PhysicsAwareBase):
    """
    Physics-Aware Neural Network (MVP):

    A generic supervised model with an optional physics regularizer hook.

    You provide:
      - backbone: an nn.Module mapping inputs -> outputs
      - physics_loss_fn(pred, batch_dict) -> scalar tensor

    This plays nicely with your PINNFactory / UPD:
      - physics_loss_fn can call symbolic residuals using autodiff.
    """
    def __init__(
        self,
        backbone: nn.Module,
        *,
        physics_loss_fn: Optional[Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]] = None,
        physics_weight: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = float(physics_weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
        return_loss: bool = False,
    ) -> PhysicsAwareOutput:
        y = self.backbone(x)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["data"] = self.mse(y, y_true)
            losses["total"] = losses["data"]

        if self.physics_loss_fn is not None and batch is not None:
            pl = self.physics_loss_fn(y, batch)
            losses["physics"] = pl
            losses["total"] = losses["total"] + self.physics_weight * pl

        return PhysicsAwareOutput(y=y, losses=losses, extras={})
