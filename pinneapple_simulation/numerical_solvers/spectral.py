"""Spectral solvers (MVP).

Includes:
  - Poisson equation on periodic 2D domain using FFT.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def poisson_periodic_fft(f: torch.Tensor, *, Lx: float = 1.0, Ly: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Solve ∇²u = f with periodic BC on 2D domain via FFT.

    f: (H,W)
    """
    H, W = f.shape
    fx = torch.fft.fftfreq(W, d=Lx / W).to(f.device)
    fy = torch.fft.fftfreq(H, d=Ly / H).to(f.device)
    kx2 = (2 * torch.pi * fx) ** 2
    ky2 = (2 * torch.pi * fy) ** 2
    KX2, KY2 = torch.meshgrid(kx2, ky2, indexing="xy")
    denom = KX2 + KY2
    f_hat = torch.fft.fft2(f)
    u_hat = torch.zeros_like(f_hat)
    u_hat[denom > eps] = -f_hat[denom > eps] / denom[denom > eps]
    u = torch.real(torch.fft.ifft2(u_hat))
    return u


@SolverRegistry.register(
    name="spectral",
    family="pde",
    description="Spectral FFT solvers (Poisson periodic 2D MVP).",
    tags=["spectral", "fft", "pde"],
)
class SpectralSolver(SolverBase):
    def forward(self, f: torch.Tensor, *, Lx: float = 1.0, Ly: float = 1.0) -> SolverOutput:
        u = poisson_periodic_fft(f, Lx=Lx, Ly=Ly)
        return SolverOutput(result=u, losses={}, extras={"Lx": float(Lx), "Ly": float(Ly)})
