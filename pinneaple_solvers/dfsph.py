"""DFSPH: Divergence-free SPH (MVP placeholder).

Full DFSPH enforces density and divergence constraints iteratively.
This MVP provides a compatible solver that currently delegates to WCSPH.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import SolverBase, SolverOutput
from .sph import SPHSolver
from .registry import SolverRegistry


@SolverRegistry.register(
    name="dfsph",
    family="particle",
    description="DFSPH MVP placeholder (currently uses WCSPH core).",
    tags=["sph", "particles", "divergence_free"],
)
class DFSPHSolver(SolverBase):
    def __init__(self, **kwargs):
        super().__init__()
        self.core = SPHSolver(**kwargs)

    def forward(self, pos0: torch.Tensor, vel0: Optional[torch.Tensor] = None, *, dt: float = 1e-3, steps: int = 10) -> SolverOutput:
        # TODO: replace with true DFSPH.
        return self.core(pos0, vel0, dt=dt, steps=steps)
