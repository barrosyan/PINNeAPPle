"""Finite Difference Method (FDM) solvers (MVP).

Currently includes:
  - Poisson equation on a 2D grid with Dirichlet boundary conditions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def poisson_dirichlet_jacobi(
    f: torch.Tensor,
    bc: torch.Tensor,
    *,
    dx: float,
    dy: float,
    iters: int = 5000,
    omega: float = 0.9,
) -> torch.Tensor:
    """Solve ∇²u = f on grid using damped Jacobi.

    Inputs
    ------
    f: (H,W)
    bc: (H,W) boundary values, interior ignored
    """
    u = bc.clone()
    H, W = u.shape
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for _ in range(int(iters)):
        u_new = u.clone()
        u_new[1:-1, 1:-1] = (
            (u[1:-1, 2:] + u[1:-1, :-2]) * dy2
            + (u[2:, 1:-1] + u[:-2, 1:-1]) * dx2
            - f[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        u[1:-1, 1:-1] = (1 - omega) * u[1:-1, 1:-1] + omega * u_new[1:-1, 1:-1]
    return u


@SolverRegistry.register(
    name="fdm",
    family="pde",
    description="Finite Difference Method solvers (Poisson 2D MVP).",
    tags=["fdm", "pde"],
)
class FDMSolver(SolverBase):
    def __init__(self, iters: int = 5000, omega: float = 0.9):
        super().__init__()
        self.iters = int(iters)
        self.omega = float(omega)

    def forward(self, f: torch.Tensor, bc: torch.Tensor, *, dx: float = 1.0, dy: float = 1.0) -> SolverOutput:
        u = poisson_dirichlet_jacobi(f, bc, dx=dx, dy=dy, iters=self.iters, omega=self.omega)
        return SolverOutput(result=u, losses={}, extras={"iters": self.iters})
