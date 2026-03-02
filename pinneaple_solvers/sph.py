"""SPH: Weakly-compressible Smoothed Particle Hydrodynamics (WCSPH) MVP.

This is a small educational implementation to generate synthetic particle-flow
datasets and to act as a baseline.

It supports 2D/3D, uniform mass particles, and simple box boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .base import SolverBase, SolverOutput
from .neighbors import build_neighbor_list
from .sph_boundaries import BoxBoundary, wall_repulsion_force
from .registry import SolverRegistry


def _poly6(r: torch.Tensor, h: float) -> torch.Tensor:
    # r: (...)
    h2 = h * h
    mask = (r >= 0) & (r <= h)
    out = torch.zeros_like(r)
    c = 315.0 / (64.0 * torch.pi * (h**9))
    out[mask] = c * (h2 - r[mask] ** 2) ** 3
    return out


def _spiky_grad(r_vec: torch.Tensor, r: torch.Tensor, h: float) -> torch.Tensor:
    # r_vec: (E,D), r: (E,)
    mask = (r > 1e-12) & (r <= h)
    out = torch.zeros_like(r_vec)
    c = -45.0 / (torch.pi * (h**6))
    out[mask] = c * ((h - r[mask]) ** 2).unsqueeze(1) * (r_vec[mask] / r[mask].unsqueeze(1))
    return out


@dataclass
class SPHState:
    pos: torch.Tensor  # (N,D)
    vel: torch.Tensor  # (N,D)
    rho: torch.Tensor  # (N,)
    p: torch.Tensor  # (N,)


@SolverRegistry.register(
    name="sph",
    family="particle",
    description="WCSPH MVP: weakly compressible SPH for fluids.",
    tags=["sph", "particles", "fluids"],
)
class SPHSolver(SolverBase):
    def __init__(
        self,
        h: float = 0.04,
        mass: float = 1.0,
        rho0: float = 1000.0,
        c0: float = 20.0,
        mu: float = 0.02,
        gravity: Optional[list[float]] = None,
        boundary_lo: Optional[list[float]] = None,
        boundary_hi: Optional[list[float]] = None,
    ):
        super().__init__()
        self.h = float(h)
        self.mass = float(mass)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.mu = float(mu)
        self.gravity = gravity
        self.boundary_lo = boundary_lo
        self.boundary_hi = boundary_hi

    def step(self, state: SPHState, dt: float) -> SPHState:
        pos, vel = state.pos, state.vel
        N, D = pos.shape
        device = pos.device

        # neighbors
        nl = build_neighbor_list(pos, h=self.h)

        # density
        rho = torch.full((N,), self.mass * _poly6(torch.zeros(1, device=device), self.h).item(), device=device)
        # accumulate
        # iterate CSR
        for i in range(N):
            js = nl.j[nl.indptr[i] : nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij = pos[i].unsqueeze(0) - pos[js]
            r = torch.norm(rij, dim=1)
            rho[i] = rho[i] + self.mass * torch.sum(_poly6(r, self.h))

        # equation of state (Tait)
        gamma = 7.0
        p = self.c0**2 * self.rho0 / gamma * ((rho / self.rho0) ** gamma - 1.0)

        # forces
        f = torch.zeros_like(pos)
        g = torch.tensor(self.gravity if self.gravity is not None else [0.0] * (D - 1) + [-9.81], device=device, dtype=pos.dtype)
        f = f + rho[:, None] * g[None, :]

        for i in range(N):
            js = nl.j[nl.indptr[i] : nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij = pos[i].unsqueeze(0) - pos[js]
            vij = vel[i].unsqueeze(0) - vel[js]
            r = torch.norm(rij, dim=1)
            gradW = _spiky_grad(rij, r, self.h)
            # pressure
            f_p = -self.mass * torch.sum((p[i] / (rho[i] ** 2) + p[js] / (rho[js] ** 2))[:, None] * gradW, dim=0)
            # viscosity (simple)
            f_v = self.mu * self.mass * torch.sum(vij, dim=0)
            f[i] = f[i] + f_p + f_v

        # boundaries
        if self.boundary_lo is not None and self.boundary_hi is not None:
            bd = BoxBoundary.from_bounds(self.boundary_lo, self.boundary_hi, device=device, dtype=pos.dtype)
            f = f + wall_repulsion_force(pos, vel, bd)

        # integrate (symplectic Euler)
        vel_new = vel + dt * (f / (rho[:, None] + 1e-12))
        pos_new = pos + dt * vel_new
        return SPHState(pos=pos_new, vel=vel_new, rho=rho, p=p)

    def forward(self, pos0: torch.Tensor, vel0: Optional[torch.Tensor] = None, *, dt: float = 1e-3, steps: int = 10) -> SolverOutput:
        """Simulate.

        Inputs
        ------
        pos0: (N,D)
        vel0: (N,D) optional (defaults zeros)

        Returns
        -------
        result: (steps+1, N, D) positions
        extras: velocities, densities, pressures
        """
        if vel0 is None:
            vel0 = torch.zeros_like(pos0)
        state = SPHState(pos=pos0, vel=vel0, rho=torch.zeros(pos0.shape[0], device=pos0.device), p=torch.zeros(pos0.shape[0], device=pos0.device))

        traj = [state.pos]
        vtraj = [state.vel]
        for _ in range(int(steps)):
            state = self.step(state, dt=float(dt))
            traj.append(state.pos)
            vtraj.append(state.vel)

        result = torch.stack(traj, dim=0)
        return SolverOutput(result=result, losses={}, extras={"vel": torch.stack(vtraj, dim=0), "rho": state.rho, "p": state.p})
