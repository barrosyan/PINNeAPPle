"""Material Point Method (MPM) for soft bodies and fluids.

A differentiable MPM implementation in pure PyTorch, following the
MLS-MPM (Moving Least Squares) formulation of Hu et al. (2018).

Supported materials
-------------------
- ``"elastic"``  – Neo-Hookean elastic solid (snow / sand approximation)
- ``"fluid"``    – Weakly-compressible Newtonian fluid (water)
- ``"snow"``     – Drucker-Prager plasticity model

All operations use standard PyTorch, enabling gradients to flow through
the entire simulation for inverse problems and differentiable physics.

Reference
---------
Hu et al., "A Moving Least Squares Material Point Method with Displacement
Discontinuity and Two-Way Rigid Body Coupling", SIGGRAPH 2018.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MPM State
# ---------------------------------------------------------------------------


class MPMState:
    """Container for MPM particle and grid state.

    Particle quantities (per particle):
    - ``pos``     : ``(n_p, dim)``   – particle positions
    - ``vel``     : ``(n_p, dim)``   – particle velocities
    - ``F``       : ``(n_p, dim, dim)`` – deformation gradient
    - ``C``       : ``(n_p, dim, dim)`` – affine velocity field (APIC)
    - ``Jp``      : ``(n_p,)``       – plastic deformation determinant

    Grid quantities (computed and discarded each step):
    - ``grid_v``  : ``(res^dim, dim)`` – grid momentum / velocity
    - ``grid_m``  : ``(res^dim,)``     – grid mass
    """

    def __init__(
        self,
        pos: torch.Tensor,
        vel: Optional[torch.Tensor] = None,
        F: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        Jp: Optional[torch.Tensor] = None,
    ) -> None:
        n_p, dim = pos.shape
        device = pos.device
        dtype = pos.dtype

        self.pos = pos
        self.vel = vel if vel is not None else torch.zeros_like(pos)
        self.F = F if F is not None else \
            torch.eye(dim, device=device, dtype=dtype).unsqueeze(0).expand(n_p, -1, -1).clone()
        self.C = C if C is not None else torch.zeros(n_p, dim, dim, device=device, dtype=dtype)
        self.Jp = Jp if Jp is not None else torch.ones(n_p, device=device, dtype=dtype)

    def clone(self) -> "MPMState":
        return MPMState(
            self.pos.clone(),
            self.vel.clone(),
            self.F.clone(),
            self.C.clone(),
            self.Jp.clone(),
        )

    def n_particles(self) -> int:
        return self.pos.shape[0]

    def dim(self) -> int:
        return self.pos.shape[1]


# ---------------------------------------------------------------------------
# MPM Simulator
# ---------------------------------------------------------------------------


class MPMSimulator(nn.Module):
    """Differentiable Material Point Method (MPM) simulator.

    Implements the MLS-MPM algorithm with quadratic B-spline kernel
    on a uniform Cartesian grid.

    Parameters
    ----------
    grid_resolution:
        Number of grid cells along each dimension (e.g. 64 → 64×64 in 2-D).
    dt:
        Time step in seconds.
    material:
        One of ``"elastic"``, ``"fluid"``, or ``"snow"``.
    dim:
        Spatial dimension (2 or 3).  Note: 3-D is memory-intensive.
    E:
        Young's modulus.
    nu:
        Poisson's ratio.
    rho:
        Reference density (kg/m³ or non-dimensionalised).
    gravity:
        Gravitational acceleration vector ``(dim,)``.
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        dt: float = 1e-4,
        material: str = "elastic",
        dim: int = 2,
        E: float = 1e4,
        nu: float = 0.2,
        rho: float = 1.0,
        gravity: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.res = grid_resolution
        self.dt = dt
        self.material = material
        self.dim = dim
        self.rho = rho

        # Lame parameters
        self.mu_0 = E / (2.0 * (1.0 + nu))
        self.lam_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Gravity
        if gravity is None:
            g = [0.0] * dim
            if dim >= 2:
                g[1] = -9.81
        else:
            g = list(gravity)[:dim]
        self.register_buffer("gravity", torch.tensor(g, dtype=torch.float32))

        # Volume per particle initialised during first forward call
        self._vol: Optional[float] = None
        self._dx: float = 1.0 / grid_resolution

    # ------------------------------------------------------------------
    # Constitutive model
    # ------------------------------------------------------------------

    def _stress(
        self, F: torch.Tensor, Jp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Kirchhoff stress P for the configured material.

        Parameters
        ----------
        F : ``(n_p, d, d)``  –  deformation gradient
        Jp : ``(n_p,)``      –  plastic deformation ratio

        Returns
        -------
        P : ``(n_p, d, d)``   –  first Piola-Kirchhoff stress
        F_new : same shape as F  –  updated deformation gradient (after projection)
        """
        n_p, d, _ = F.shape
        mu = self.mu_0
        lam = self.lam_0

        J = torch.linalg.det(F).clamp(min=1e-8)  # (n_p,)

        if self.material == "fluid":
            # Weakly compressible: P = (J - 1) * lam * I
            P = lam * (J - 1.0).view(n_p, 1, 1) * \
                torch.eye(d, device=F.device, dtype=F.dtype).unsqueeze(0)
            F_new = F
        elif self.material == "snow":
            # Drucker-Prager: SVD-based clamping of singular values
            try:
                U, sig, Vh = torch.linalg.svd(F)
                sig_clamp = sig.clamp(1.0 - 2.5e-2, 1.0 + 4.5e-3)
                F_proj = U @ torch.diag_embed(sig_clamp) @ Vh
                # Neo-Hookean on projected deformation
                J_e = torch.linalg.det(F_proj).clamp(min=1e-8)
                F_T_inv = torch.linalg.inv(F_proj).mT
                P = mu * (F_proj - F_T_inv) + lam * (J_e - 1.0).view(n_p, 1, 1) * F_T_inv
                F_new = F_proj
            except Exception:
                # Fallback: elastic model
                J_e = J
                F_T_inv = torch.linalg.inv(F).mT
                P = mu * (F - F_T_inv) + lam * (J_e - 1.0).view(n_p, 1, 1) * F_T_inv
                F_new = F
        else:
            # Neo-Hookean elastic solid
            F_T_inv = torch.linalg.inv(F).mT      # (n_p, d, d)
            P = mu * (F - F_T_inv) + \
                lam * (J - 1.0).view(n_p, 1, 1) * F_T_inv
            F_new = F

        return P, F_new

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _grid_shape(self) -> Tuple[int, ...]:
        return tuple([self.res] * self.dim)

    def _base_and_weights(
        self, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute grid base cell and quadratic B-spline weights.

        Parameters
        ----------
        pos : ``(n_p, dim)``  –  particle positions in [0, 1]^dim

        Returns
        -------
        base : ``(n_p, dim)``   –  integer base cell indices
        w    : ``(n_p, dim, 3)`` –  quadratic B-spline weights for 3 cells
        """
        dx = self._dx
        xp = pos / dx - 0.5                # fractional grid position
        base = xp.long()                   # (n_p, dim)
        fx = xp - base.float()            # (n_p, dim) ∈ [0, 1)
        # Quadratic B-spline
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        w = torch.stack([w0, w1, w2], dim=-1)  # (n_p, dim, 3)
        return base, w

    # ------------------------------------------------------------------
    # P2G: particle-to-grid
    # ------------------------------------------------------------------

    def p2g(self, state: MPMState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transfer particle data to the Eulerian grid.

        Returns
        -------
        grid_momentum : ``(res^dim, dim)``
        grid_mass     : ``(res^dim,)``
        """
        n_p = state.n_particles()
        d = self.dim
        res = self.res
        dx = self._dx
        dt = self.dt

        if self._vol is None:
            self._vol = (dx ** d) * 0.5  # approximate particle volume

        vol = self._vol
        P, F_new = self._stress(state.F, state.Jp)
        # Update deformation gradient
        state.F = F_new

        # Affine contribution to stress: -vol * P * F^T * 4 / dx^2
        stress = -dt * vol * (4.0 / dx ** 2) * P @ state.F.mT  # (n_p, d, d)
        # APIC affine velocity matrix
        affine = stress + self.rho * vol * state.C               # (n_p, d, d)

        base, w = self._base_and_weights(state.pos)              # (n_p, d), (n_p, d, 3)

        # Flat grid tensors
        flat_sz = res ** d
        grid_mom = torch.zeros(flat_sz, d, device=state.pos.device, dtype=state.pos.dtype)
        grid_mass = torch.zeros(flat_sz, device=state.pos.device, dtype=state.pos.dtype)

        # Scatter (loop over 3^d stencil nodes)
        import itertools
        for offsets in itertools.product(range(3), repeat=d):
            off = torch.tensor(offsets, device=state.pos.device)
            node = (base + off.unsqueeze(0)).clamp(0, res - 1)   # (n_p, d)

            # Weight product over dimensions
            ww = torch.ones(n_p, device=state.pos.device, dtype=state.pos.dtype)
            for dim_i in range(d):
                ww = ww * w[:, dim_i, offsets[dim_i]]            # (n_p,)

            # Displacement from particle to node
            dpos = (off.float().unsqueeze(0) - (state.pos / dx - 0.5 - base.float())) * dx
            # (n_p, d)

            # Flat node indices
            flat_idx = _flat_index(node, res, d)                 # (n_p,)
            mass_contrib = ww * self.rho * vol               # (n_p,)
            mom_contrib = ww.unsqueeze(-1) * (
                self.rho * vol * state.vel
                + (affine @ dpos.unsqueeze(-1)).squeeze(-1)
            )

            grid_mass.scatter_add_(0, flat_idx, mass_contrib)
            grid_mom.scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, d), mom_contrib)

        return grid_mom, grid_mass

    # ------------------------------------------------------------------
    # Grid update: apply forces and boundary conditions
    # ------------------------------------------------------------------

    def grid_update(
        self, grid_mass: torch.Tensor, grid_momentum: torch.Tensor
    ) -> torch.Tensor:
        """Apply gravity and enforce boundary conditions on the grid.

        Parameters
        ----------
        grid_mass : ``(flat_sz,)``
        grid_momentum : ``(flat_sz, dim)``

        Returns
        -------
        grid_velocity : ``(flat_sz, dim)``
        """
        res = self.res
        d = self.dim
        dt = self.dt

        mask = grid_mass > 1e-10
        grid_vel = torch.zeros_like(grid_momentum)
        grid_vel[mask] = grid_momentum[mask] / grid_mass[mask].unsqueeze(-1)

        # Apply gravity
        grid_vel = grid_vel + dt * self.gravity.to(grid_vel.device).unsqueeze(0)

        # Enforce sticky walls (zero-velocity Dirichlet BC on boundary cells)
        boundary = 3
        flat_sz = res ** d
        grid_vel = self._apply_bc(grid_vel, res, d, boundary)

        return grid_vel

    def _apply_bc(
        self, grid_vel: torch.Tensor, res: int, d: int, boundary: int
    ) -> torch.Tensor:
        """Zero-out velocities on boundary cells (sticky wall BC)."""
        import itertools
        flat_sz = res ** d
        # Build a mask of boundary cells
        bc_mask = torch.zeros(flat_sz, dtype=torch.bool, device=grid_vel.device)
        coords = torch.arange(res, device=grid_vel.device)
        grids = torch.meshgrid(*[coords] * d, indexing="ij")
        for g in grids:
            bc_mask |= (g.reshape(-1) < boundary) | (g.reshape(-1) >= res - boundary)
        grid_vel[bc_mask] = 0.0
        return grid_vel

    # ------------------------------------------------------------------
    # G2P: grid-to-particle
    # ------------------------------------------------------------------

    def g2p(self, state: MPMState, grid_velocity: torch.Tensor) -> MPMState:
        """Transfer grid velocity back to particles and update positions.

        Parameters
        ----------
        state : current particle state
        grid_velocity : ``(flat_sz, dim)`` grid velocity field

        Returns
        -------
        MPMState  –  updated particle state
        """
        n_p = state.n_particles()
        d = self.dim
        res = self.res
        dx = self.dt

        base, w = self._base_and_weights(state.pos)
        new_vel = torch.zeros_like(state.vel)
        new_C = torch.zeros_like(state.C)

        import itertools
        for offsets in itertools.product(range(3), repeat=d):
            off = torch.tensor(offsets, device=state.pos.device)
            node = (base + off.unsqueeze(0)).clamp(0, res - 1)
            flat_idx = _flat_index(node, res, d)               # (n_p,)
            g_vel = grid_velocity[flat_idx]                    # (n_p, d)

            ww = torch.ones(n_p, device=state.pos.device, dtype=state.pos.dtype)
            for dim_i in range(d):
                ww = ww * w[:, dim_i, offsets[dim_i]]

            new_vel = new_vel + ww.unsqueeze(-1) * g_vel

            dpos = (off.float().unsqueeze(0) - (state.pos / self._dx - 0.5 - base.float())) \
                   * self._dx
            new_C = new_C + (4.0 / self._dx ** 2) * ww.unsqueeze(-1).unsqueeze(-1) * \
                    g_vel.unsqueeze(-1) * dpos.unsqueeze(-2)

        new_state = state.clone()
        new_state.vel = new_vel
        new_state.C = new_C

        # Update deformation gradient: F = (I + dt * C) * F
        I = torch.eye(d, device=state.pos.device, dtype=state.pos.dtype).unsqueeze(0)
        new_state.F = (I + self.dt * new_C) @ state.F

        # Update positions
        new_state.pos = (state.pos + self.dt * new_vel).clamp(
            self._dx * 3, 1.0 - self._dx * 3
        )

        return new_state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state: MPMState, n_steps: int = 1) -> MPMState:
        """Advance the MPM simulation by *n_steps* time steps.

        Parameters
        ----------
        state : initial :class:`MPMState`
        n_steps : number of time steps to advance

        Returns
        -------
        MPMState  –  state after *n_steps* steps
        """
        for _ in range(n_steps):
            grid_mom, grid_mass = self.p2g(state)
            grid_vel = self.grid_update(grid_mass, grid_mom)
            state = self.g2p(state, grid_vel)
        return state


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _flat_index(node: torch.Tensor, res: int, d: int) -> torch.Tensor:
    """Convert ``(n_p, d)`` grid indices to flat indices in a ``res^d`` array."""
    idx = node[:, 0]
    for dim_i in range(1, d):
        idx = idx * res + node[:, dim_i]
    return idx.clamp(0, res ** d - 1)
