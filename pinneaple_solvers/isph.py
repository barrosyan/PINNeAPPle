"""ISPH: Incompressible SPH (MVP placeholder).

Full ISPH requires solving a pressure Poisson equation each step.
This MVP wraps WCSPH and exposes an interface-compatible solver so you can plug
it into the Arena and swap implementations later.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import SolverBase, SolverOutput
from .sph import SPHSolver
from .registry import SolverRegistry


@SolverRegistry.register(
    name="isph",
    family="particle",
    description="ISPH MVP placeholder (currently uses WCSPH core).",
    tags=["sph", "particles", "incompressible"],
)
class ISPHSolver(SolverBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = SPHSolver(**kwargs)

    def forward(self, pos0: torch.Tensor, vel0: Optional[torch.Tensor] = None, *, dt: float = 1e-3, steps: int = 10) -> SolverOutput:
        # TODO: replace with true ISPH (pressure projection).
        return self.core(pos0, vel0, dt=dt, steps=steps)
