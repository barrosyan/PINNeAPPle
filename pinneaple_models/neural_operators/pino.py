from __future__ import annotations
"""Physics-informed neural operator for PDE solution learning."""
from typing import Callable, Dict, Any

import torch

from .base import NeuralOperatorBase, OperatorOutput


class PhysicsInformedNeuralOperator(NeuralOperatorBase):
    """
    Wrapper adding physics loss to any operator.
    """
    def __init__(self, operator: NeuralOperatorBase):
        super().__init__()
        self.operator = operator

    def forward(
        self,
        *args,
        physics_fn: Callable[..., Dict[str, torch.Tensor]] = None,
        physics_data: Dict[str, Any] = None,
        **kw,
    ) -> OperatorOutput:
        out = self.operator(*args, **kw)
        losses = dict(out.losses)

        if physics_fn is not None and physics_data is not None:
            pl = physics_fn(out.y, **physics_data)
            losses.update(pl)
            losses["total"] = losses.get("total", 0.0) + pl.get("physics", 0.0)

        return OperatorOutput(y=out.y, losses=losses, extras=out.extras)
