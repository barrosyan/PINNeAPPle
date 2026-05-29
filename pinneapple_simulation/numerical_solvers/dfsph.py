"""DFSPH: Divergence-Free SPH (Bender & Koschier 2015 / 2017).

DFSPH enforces two constraints per time step:
  1. Divergence-Free  : ∇·v = 0  (velocity divergence solver — inner loop)
  2. Constant Density : ρ = ρ₀   (density solver — outer loop, optional)

Algorithm per step:
  1. Find neighbors, compute densities.
  2. Compute non-pressure forces (viscosity, gravity).
  3. Divergence-free pressure solve → adjust velocity so ∇·v ≈ 0.
  4. Advect positions.
  5. Density-correction pressure solve → adjust velocity so ρ ≈ ρ₀.
  6. Enforce boundary repulsion.

References
----------
Bender & Koschier (2015), SCA — DFSPH.
Bender & Koschier (2017), IEEE TVCG — improved formulation.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import SolverBase, SolverOutput
from .sph import SPHState, _poly6, _spiky_grad
from .neighbors import build_neighbor_list
from .isph import (
    _compute_density, _compute_divergence,
    _compute_pressure_gradient, _sph_laplacian_kernel,
)
from .registry import SolverRegistry


def _dfsph_alpha(pos: torch.Tensor, rho: torch.Tensor, mass: float, h: float, nl) -> torch.Tensor:
    """
    Per-particle DFSPH factor α_i = ρ_i / (∑_j m_j ∇W_ij)²  + ε.
    Used to scale the pressure correction from density/divergence error.
    """
    N, D = pos.shape
    device = pos.device
    alpha  = torch.zeros(N, device=device)

    for i in range(N):
        js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
        if js.numel() == 0:
            alpha[i] = 1.0
            continue
        rij    = pos[i].unsqueeze(0) - pos[js]
        r      = torch.norm(rij, dim=1)
        grad_w = _spiky_grad(rij, r, h)               # (E, D)
        # ∑_j m_j ∇W_ij  (D-vector)
        sum_grad = (mass * grad_w).sum(0)              # (D,)
        denom = (sum_grad ** 2).sum() + 1e-20
        alpha[i] = rho[i] / denom
    return alpha


def _viscous_force(pos: torch.Tensor, vel: torch.Tensor, rho: torch.Tensor,
                   mass: float, mu: float, h: float, nl) -> torch.Tensor:
    """Morris et al. (1997) SPH viscosity."""
    N, D   = pos.shape
    device = pos.device
    f_visc = torch.zeros(N, D, device=device)
    for i in range(N):
        js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
        if js.numel() == 0:
            continue
        rij    = pos[i].unsqueeze(0) - pos[js]
        r      = torch.norm(rij, dim=1).clamp(min=1e-12)
        grad_w = _spiky_grad(rij, r, h)
        dv     = vel[i].unsqueeze(0) - vel[js]
        coeff  = mass * 2.0 * mu / (rho[i] * rho[js] * r ** 2 + 1e-20)
        dot_r  = (dv * rij).sum(dim=1)
        f_visc[i] = (coeff.unsqueeze(1) * dot_r.unsqueeze(1) * grad_w).sum(0)
    return f_visc


def _divergence_free_solve(
    pos: torch.Tensor,
    vel: torch.Tensor,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    mass: float,
    h: float,
    nl,
    dt: float,
    max_iters: int = 5,
    tol: float = 1e-3,
) -> torch.Tensor:
    """
    Iteratively correct velocity so ∇·v ≈ 0.
    Returns corrected velocity.
    """
    N, D   = vel.shape
    device = vel.device
    v_corr = vel.clone()

    for _ in range(max_iters):
        div_v = _compute_divergence(pos, v_corr, rho, mass, h, nl)
        max_err = div_v.abs().max().item()
        if max_err < tol:
            break
        # Pressure correction: κ_i = div_v_i * α_i / dt
        kappa = div_v * alpha / dt                    # (N,)
        dp = torch.zeros(N, D, device=device)
        for i in range(N):
            js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij    = pos[i].unsqueeze(0) - pos[js]
            r      = torch.norm(rij, dim=1)
            grad_w = _spiky_grad(rij, r, h)
            coeff  = mass * (kappa[i] / rho[i] + kappa[js] / rho[js])
            dp[i]  = -(coeff.unsqueeze(1) * grad_w).sum(0)
        v_corr = v_corr + dt * dp

    return v_corr


def _density_solve(
    pos: torch.Tensor,
    vel: torch.Tensor,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    rho0: float,
    mass: float,
    h: float,
    nl,
    dt: float,
    max_iters: int = 5,
    tol: float = 0.01,
) -> torch.Tensor:
    """
    Iteratively correct velocity so ρ ≈ ρ₀.
    Returns corrected velocity.
    """
    N, D   = vel.shape
    device = vel.device
    v_corr = vel.clone()

    for _ in range(max_iters):
        # Predicted density after one step
        div_v  = _compute_divergence(pos, v_corr, rho, mass, h, nl)
        rho_pred = rho + dt * rho * div_v
        err      = (rho_pred - rho0) / rho0
        max_err  = err.abs().max().item()
        if max_err < tol:
            break
        kappa  = err * rho / dt * alpha               # (N,)
        dp     = torch.zeros(N, D, device=device)
        for i in range(N):
            js = nl.j[nl.indptr[i]: nl.indptr[i + 1]]
            if js.numel() == 0:
                continue
            rij    = pos[i].unsqueeze(0) - pos[js]
            r      = torch.norm(rij, dim=1)
            grad_w = _spiky_grad(rij, r, h)
            coeff  = mass * (kappa[i] / rho[i] + kappa[js] / rho[js])
            dp[i]  = -(coeff.unsqueeze(1) * grad_w).sum(0)
        v_corr = v_corr + dt * dp

    return v_corr


@SolverRegistry.register(
    name="dfsph",
    family="particle",
    description="DFSPH: Divergence-Free SPH (Bender & Koschier 2015) with density and divergence solvers.",
    tags=["sph", "particles", "divergence_free", "incompressible"],
)
class DFSPHSolver(SolverBase):
    """
    Divergence-Free SPH (DFSPH) solver.

    Parameters
    ----------
    h               : smoothing length
    mass            : particle mass
    rho0            : reference density
    mu              : dynamic viscosity
    gravity         : body acceleration list  e.g. [0.0, -9.81]
    boundary_lo/hi  : domain extents for wall repulsion
    div_max_iters   : divergence-free inner loop iterations
    dens_max_iters  : density-correction inner loop iterations
    enable_density_solve : if False, only divergence-free solve is used (faster)
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
        div_max_iters: int = 5,
        dens_max_iters: int = 5,
        enable_density_solve: bool = True,
    ):
        super().__init__()
        self.h                   = float(h)
        self.mass                = float(mass)
        self.rho0                = float(rho0)
        self.mu                  = float(mu)
        self.gravity             = gravity
        self.boundary_lo         = boundary_lo
        self.boundary_hi         = boundary_hi
        self.div_max_iters       = int(div_max_iters)
        self.dens_max_iters      = int(dens_max_iters)
        self.enable_density_solve = bool(enable_density_solve)

    def step(self, state: SPHState, dt: float) -> SPHState:
        pos, vel = state.pos, state.vel
        device   = pos.device
        nl       = build_neighbor_list(pos, h=self.h)

        # ── 1. Density ─────────────────────────────────────────────────────────
        rho   = _compute_density(pos, self.mass, self.h, nl)
        alpha = _dfsph_alpha(pos, rho, self.mass, self.h, nl)

        # ── 2. Non-pressure forces ─────────────────────────────────────────────
        f_visc = _viscous_force(pos, vel, rho, self.mass, self.mu, self.h, nl)
        f_body = torch.zeros_like(vel)
        if self.gravity is not None:
            g = torch.tensor(self.gravity, dtype=torch.float32, device=device)
            f_body[:] = g
        vel_adv = vel + dt * (f_visc / rho.unsqueeze(1) + f_body)

        # ── 3. Divergence-free velocity correction ─────────────────────────────
        vel_div = _divergence_free_solve(
            pos, vel_adv, rho, alpha, self.mass, self.h, nl, dt, self.div_max_iters)

        # ── 4. Advect positions ────────────────────────────────────────────────
        pos_new = pos + dt * vel_div

        # ── 5. Density correction (optional) ──────────────────────────────────
        if self.enable_density_solve:
            nl2     = build_neighbor_list(pos_new, h=self.h)
            rho2    = _compute_density(pos_new, self.mass, self.h, nl2)
            alpha2  = _dfsph_alpha(pos_new, rho2, self.mass, self.h, nl2)
            vel_fin = _density_solve(
                pos_new, vel_div, rho2, alpha2, self.rho0,
                self.mass, self.h, nl2, dt, self.dens_max_iters)
        else:
            vel_fin = vel_div

        # ── 6. Wall repulsion ──────────────────────────────────────────────────
        if self.boundary_lo is not None and self.boundary_hi is not None:
            from .sph_boundaries import BoxBoundary, wall_repulsion_force
            box   = BoxBoundary(lo=self.boundary_lo, hi=self.boundary_hi)
            f_rep = wall_repulsion_force(pos_new, box, h=self.h, c0=10.0)
            vel_fin = vel_fin + dt * f_rep

        # dummy pressure (DFSPH does not track absolute pressure)
        p = torch.zeros(pos.shape[0], device=device)
        return SPHState(pos=pos_new, vel=vel_fin, rho=rho, p=p)

    def forward(
        self,
        pos0: torch.Tensor,
        vel0: Optional[torch.Tensor] = None,
        *,
        dt: float = 1e-3,
        steps: int = 10,
    ) -> SolverOutput:
        N      = pos0.shape[0]
        device = pos0.device
        vel    = vel0 if vel0 is not None else torch.zeros_like(pos0)
        state  = SPHState(
            pos=pos0.clone(),
            vel=vel.clone(),
            rho=torch.full((N,), self.rho0, device=device),
            p=torch.zeros(N, device=device),
        )

        traj_pos, traj_vel = [], []
        for _ in range(steps):
            state = self.step(state, dt)
            traj_pos.append(state.pos.clone())
            traj_vel.append(state.vel.clone())

        return SolverOutput(
            result=state.pos,
            losses={"divergence": torch.zeros(1, device=device)},
            extras={
                "pos":      state.pos,
                "vel":      state.vel,
                "rho":      state.rho,
                "traj_pos": traj_pos,
                "traj_vel": traj_vel,
            },
        )
