"""ISPH: Incompressible SPH via pressure-projection (Cummins & Rudman 1999).

The solver enforces incompressibility through a prediction-correction loop:
  1. Predict   : v* = v_n + dt * (f_viscous + f_body) / ρ  (no pressure)
  2. Solve PPE : ∇²p = ρ/dt · ∇·v*  (Gauss-Seidel on SPH Laplacian)
  3. Correct   : v_(n+1) = v* − dt/ρ · ∇p
  4. Advect    : x_(n+1) = x_n + dt · v_(n+1)

The PPE is solved using the SPH Laplacian discretisation of Shao & Lo (2003):
    ∑_j m_j/ρ_j · (p_i − p_j) · L_ij = ρ_i/dt · ∇·v*_i

where L_ij = 2·|∇W_ij|² / (r_ij² + ε²).

References
----------
Cummins & Rudman (1999), JCP 152, 584-607.
Shao & Lo (2003), Advances in Water Resources 26, 787-800.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import SolverBase, SolverOutput
from .sph import SPHState, _poly6, _spiky_grad
from .neighbors import build_neighbor_list
from .sph_boundaries import wall_repulsion_force
from .registry import SolverRegistry


def _sph_laplacian_kernel(r_vec: torch.Tensor, r: torch.Tensor, h: float) -> torch.Tensor:
    """
    SPH Laplacian kernel coefficient L_ij = 2|∇W|² / (r² + ε²).
    Used to discretise ∇·(1/ρ ∇p) in the PPE.
    r_vec: (E, D), r: (E,)  →  L_ij: (E,)
    """
    mask = (r > 1e-12) & (r <= h)
    c6   = 45.0 / (torch.pi * h ** 6)    # spiky grad magnitude prefactor
    grad_mag = torch.zeros_like(r)
    if mask.any():
        grad_mag[mask] = c6 * (h - r[mask]) ** 2
    eps2 = (0.01 * h) ** 2
    return 2.0 * grad_mag ** 2 / (r ** 2 + eps2)


def _compute_density(pos: torch.Tensor, mass: float, h: float, nl) -> torch.Tensor:
    """SPH density summation."""
    N, D = pos.shape
    device = pos.device
    rho = torch.full((N,), mass * _poly6(torch.zeros(1, device=device), h).item(), device=device)
    for i in range(N):
        js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
        if js.numel() == 0:
            continue
        rij = pos[i].unsqueeze(0) - pos[js]
        r = torch.norm(rij, dim=1)
        rho[i] = rho[i] + mass * torch.sum(_poly6(r, h))
    return rho.clamp(min=1e-3)


def _compute_divergence(pos: torch.Tensor, vel: torch.Tensor, rho: torch.Tensor,
                         mass: float, h: float, nl) -> torch.Tensor:
    """SPH divergence:  ∇·v_i = ∑_j m_j/ρ_j · (v_j − v_i) · ∇W_ij."""
    N = pos.shape[0]
    device = pos.device
    div_v = torch.zeros(N, device=device)
    for i in range(N):
        js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
        if js.numel() == 0:
            continue
        rij    = pos[i].unsqueeze(0) - pos[js]         # (E, D)
        r      = torch.norm(rij, dim=1)                 # (E,)
        grad_w = _spiky_grad(rij, r, h)                 # (E, D)
        dv     = vel[js] - vel[i].unsqueeze(0)          # (E, D)
        div_v[i] = torch.sum((mass / rho[js]) * (dv * grad_w).sum(dim=1))
    return div_v


def _compute_pressure_gradient(pos: torch.Tensor, p: torch.Tensor, rho: torch.Tensor,
                                 mass: float, h: float, nl) -> torch.Tensor:
    """SPH pressure gradient: ∇p_i = ρ_i ∑_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij."""
    N, D = pos.shape
    device = pos.device
    grad_p = torch.zeros(N, D, device=device)
    for i in range(N):
        js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
        if js.numel() == 0:
            continue
        rij    = pos[i].unsqueeze(0) - pos[js]
        r      = torch.norm(rij, dim=1)
        grad_w = _spiky_grad(rij, r, h)                 # (E, D)
        coeff  = mass * (p[i] / rho[i] ** 2 + p[js] / rho[js] ** 2)   # (E,)
        grad_p[i] = rho[i] * (coeff.unsqueeze(1) * grad_w).sum(0)
    return grad_p


def _solve_ppe(pos: torch.Tensor, rho: torch.Tensor, mass: float, h: float, nl,
               rhs: torch.Tensor, iters: int = 30) -> torch.Tensor:
    """
    Solve the PPE:  ∑_j m_j/ρ_j · (p_i − p_j) · L_ij = rhs_i
    via Gauss-Seidel iterations.  L_ij is the SPH Laplacian kernel.
    """
    N = pos.shape[0]
    device = pos.device
    p = torch.zeros(N, device=device)

    for _ in range(iters):
        p_new = p.clone()
        for i in range(N):
            js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij     = pos[i].unsqueeze(0) - pos[js]
            r       = torch.norm(rij, dim=1)
            L_ij    = _sph_laplacian_kernel(rij, r, h)           # (E,)
            w_ij    = mass / rho[js] * L_ij                      # (E,)
            diag    = w_ij.sum()
            if diag.abs() < 1e-20:
                continue
            off_sum = (w_ij * p[js]).sum()
            p_new[i] = (rhs[i] + off_sum) / diag
        p = p_new

    return p


@SolverRegistry.register(
    name="isph",
    family="particle",
    description="ISPH: Incompressible SPH via pressure-projection (Cummins & Rudman 1999).",
    tags=["sph", "particles", "incompressible"],
)
class ISPHSolver(SolverBase):
    """
    Incompressible SPH (ISPH) using prediction-correction pressure projection.

    Parameters
    ----------
    h         : smoothing length
    mass      : particle mass (all equal)
    rho0      : reference density
    mu        : dynamic viscosity
    gravity   : body-force acceleration list  [gx, gy] or [gx, gy, gz]
    boundary_lo, boundary_hi : domain extents for wall repulsion
    ppe_iters : Gauss-Seidel iterations per pressure solve (default 30)
    """

    def __init__(
        self,
        h: float = 0.04,
        mass: float = 1.0,
        rho0: float = 1000.0,
        mu: float = 0.02,
        gravity: Optional[list] = None,
        boundary_lo: Optional[list] = None,
        boundary_hi: Optional[list] = None,
        ppe_iters: int = 30,
    ):
        super().__init__()
        self.h          = float(h)
        self.mass       = float(mass)
        self.rho0       = float(rho0)
        self.mu         = float(mu)
        self.gravity    = gravity
        self.boundary_lo = boundary_lo
        self.boundary_hi = boundary_hi
        self.ppe_iters  = int(ppe_iters)

    def _viscous_force(self, pos: torch.Tensor, vel: torch.Tensor, rho: torch.Tensor, nl) -> torch.Tensor:
        """SPH viscous force via Morris et al. (1997) formulation."""
        N, D = pos.shape
        device = pos.device
        f_visc = torch.zeros(N, D, device=device)
        for i in range(N):
            js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij    = pos[i].unsqueeze(0) - pos[js]
            r      = torch.norm(rij, dim=1).clamp(min=1e-12)
            grad_w = _spiky_grad(rij, r, h=self.h)             # (E, D)
            dv     = vel[i].unsqueeze(0) - vel[js]             # (E, D)
            mu_ij  = 2.0 * self.mu
            coeff  = self.mass * mu_ij / (rho[i] * rho[js] * r ** 2 + 1e-20)
            dot_r  = (dv * rij).sum(dim=1)                     # (E,)
            f_visc[i] = (coeff.unsqueeze(1) * dot_r.unsqueeze(1) * grad_w).sum(0)
        return f_visc

    def step(self, state: SPHState, dt: float) -> SPHState:
        pos, vel = state.pos, state.vel
        device   = pos.device
        nl       = build_neighbor_list(pos, h=self.h)

        # ── 1. Density ─────────────────────────────────────────────────────────
        rho = _compute_density(pos, self.mass, self.h, nl)

        # ── 2. Predict velocity (no pressure) ─────────────────────────────────
        f_visc = self._viscous_force(pos, vel, rho, nl)
        f_body = torch.zeros_like(vel)
        if self.gravity is not None:
            g = torch.tensor(self.gravity, dtype=torch.float32, device=device)
            f_body[:] = g
        a_pred   = f_visc / rho.unsqueeze(1) + f_body
        vel_star = vel + dt * a_pred

        # ── 3. Solve PPE: ∇²p = ρ/dt · ∇·v*  ─────────────────────────────────
        div_v_star = _compute_divergence(pos, vel_star, rho, self.mass, self.h, nl)
        rhs        = rho / dt * div_v_star
        p          = _solve_ppe(pos, rho, self.mass, self.h, nl, rhs, self.ppe_iters)

        # ── 4. Velocity correction ─────────────────────────────────────────────
        grad_p  = _compute_pressure_gradient(pos, p, rho, self.mass, self.h, nl)
        vel_new = vel_star - dt / rho.unsqueeze(1) * grad_p

        # ── 5. Wall repulsion ──────────────────────────────────────────────────
        if self.boundary_lo is not None and self.boundary_hi is not None:
            from .sph_boundaries import BoxBoundary, wall_repulsion_force
            box   = BoxBoundary(lo=self.boundary_lo, hi=self.boundary_hi)
            f_rep = wall_repulsion_force(pos, box, h=self.h, c0=10.0)
            vel_new = vel_new + dt / rho.unsqueeze(1) * f_rep

        # ── 6. Advect positions ────────────────────────────────────────────────
        pos_new = pos + dt * vel_new

        return SPHState(pos=pos_new, vel=vel_new, rho=rho, p=p)

    def forward(
        self,
        pos0: torch.Tensor,
        vel0: Optional[torch.Tensor] = None,
        *,
        dt: float = 1e-3,
        steps: int = 10,
    ) -> SolverOutput:
        N, D  = pos0.shape
        device = pos0.device
        vel   = vel0 if vel0 is not None else torch.zeros_like(pos0)
        state = SPHState(
            pos=pos0.clone(),
            vel=vel.clone(),
            rho=torch.full((N,), self.rho0, device=device),
            p=torch.zeros(N, device=device),
        )

        traj_pos, traj_vel = [], []
        for s in range(steps):
            state = self.step(state, dt)
            traj_pos.append(state.pos.clone())
            traj_vel.append(state.vel.clone())

        return SolverOutput(
            result=state.pos,
            losses={"divergence": torch.zeros(1, device=device)},
            extras={
                "pos":       state.pos,
                "vel":       state.vel,
                "rho":       state.rho,
                "p":         state.p,
                "traj_pos":  traj_pos,
                "traj_vel":  traj_vel,
            },
        )
