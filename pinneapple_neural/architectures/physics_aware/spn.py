from __future__ import annotations
"""Structure-preserving network for constrained dynamics."""
from typing import Dict, Optional, Callable

import torch
import torch.nn as nn

from .base import PhysicsAwareBase, PhysicsAwareOutput


class StructurePreservingNetwork(PhysicsAwareBase):
    """
    Structure-Preserving Network (MVP):

    Idea:
      - Learn an unconstrained predictor y_raw
      - Apply a projection / correction operator Π that enforces structure:
          y = Π(y_raw)
    Examples:
      - non-negativity: clamp
      - unit vector normalization
      - divergence-free: projection in Fourier space (future)
      - energy conservation constraints (approx)

    In MVP, Π is user-supplied (callable).
    """
    def __init__(
        self,
        backbone: nn.Module,
        *,
        projector: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> PhysicsAwareOutput:
        y_raw = self.backbone(x)
        y = self.projector(y_raw)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return PhysicsAwareOutput(y=y, losses=losses, extras={"y_raw": y_raw})
