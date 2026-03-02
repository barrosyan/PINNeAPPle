"""LBM: Lattice Boltzmann Method (MVP D2Q9).

This is a minimal D2Q9 BGK implementation aimed at generating toy fluid
datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def _d2q9():
    c = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=torch.int64,
    )
    w = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=torch.float32)
    return c, w


def lbm_step(f: torch.Tensor, omega: float, force: Optional[torch.Tensor] = None):
    """One LBM step.

    f: (9,H,W)
    force: (2,H,W)
    """
    device = f.device
    c, w = _d2q9()
    c = c.to(device)
    w = w.to(device)

    rho = torch.sum(f, dim=0)
    ux = torch.sum(f * c[:, 0].view(-1, 1, 1), dim=0) / (rho + 1e-12)
    uy = torch.sum(f * c[:, 1].view(-1, 1, 1), dim=0) / (rho + 1e-12)
    if force is not None:
        ux = ux + 0.5 * force[0] / (rho + 1e-12)
        uy = uy + 0.5 * force[1] / (rho + 1e-12)

    u2 = ux**2 + uy**2
    feq = torch.zeros_like(f)
    for i in range(9):
        cu = 3.0 * (c[i, 0] * ux + c[i, 1] * uy)
        feq[i] = w[i] * rho * (1 + cu + 0.5 * cu**2 - 1.5 * u2)

    f = (1 - omega) * f + omega * feq

    # streaming
    f_stream = torch.zeros_like(f)
    for i in range(9):
        dx, dy = int(c[i, 0].item()), int(c[i, 1].item())
        f_stream[i] = torch.roll(f[i], shifts=(dy, dx), dims=(0, 1))
    return f_stream


@SolverRegistry.register(
    name="lbm",
    family="pde",
    description="Lattice Boltzmann D2Q9 BGK (toy).",
    tags=["lbm", "fluids"],
)
class LBMSolver(SolverBase):
    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, f0: torch.Tensor, *, steps: int = 100, force: Optional[torch.Tensor] = None) -> SolverOutput:
        f = f0
        for _ in range(int(steps)):
            f = lbm_step(f, omega=self.omega, force=force)
        return SolverOutput(result=f, losses={}, extras={"steps": int(steps)})
