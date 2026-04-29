"""Particle-based simulations: SPH (Smoothed Particle Hydrodynamics) and DEM.

All operations are implemented in pure PyTorch so that gradients flow through
the simulation for inverse problems and differentiable physics.

Smoothed Particle Hydrodynamics (SPH)
--------------------------------------
Uses the weakly-compressible SPH (WCSPH) formulation:
- Density computed from the kernel summation.
- Pressure from the Tait equation of state.
- Viscous forces via the Laminar viscosity model.
- Boundary particles (mirrored ghost particles) for wall BCs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base particle system
# ---------------------------------------------------------------------------


class ParticleSystem(nn.Module):
    """Base particle system with brute-force O(N²) neighbour search.

    For production use, replace :meth:`_find_neighbours` with a spatial hash
    or a tree structure (e.g. via the ``torch_cluster`` package).

    Parameters
    ----------
    n_particles:
        Number of particles.
    dim:
        Spatial dimension (2 or 3).
    device:
        PyTorch device.
    """

    def __init__(
        self,
        n_particles: int,
        dim: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.n = n_particles
        self.dim = dim
        self.device = device

    def _find_neighbours(
        self,
        pos: torch.Tensor,
        h: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return neighbours within smoothing length *h*.

        Parameters
        ----------
        pos : ``(N, d)``

        Returns
        -------
        i_idx, j_idx : ``(n_pairs,)``  –  particle pair indices
        r_ij         : ``(n_pairs, d)``  –  displacement vectors r_i - r_j
        """
        # Pairwise displacement (N x N x d)
        r = pos.unsqueeze(0) - pos.unsqueeze(1)         # (N, N, d)
        dist = r.norm(dim=-1)                            # (N, N)
        mask = (dist < h) & (dist > 0.0)                # exclude self
        i_idx, j_idx = torch.where(mask)
        r_ij = r[i_idx, j_idx]                          # (n_pairs, d)
        return i_idx, j_idx, r_ij

    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("Subclasses must implement forward().")


# ---------------------------------------------------------------------------
# SPH kernel
# ---------------------------------------------------------------------------


def _cubic_kernel(q: torch.Tensor, h: float, dim: int) -> torch.Tensor:
    """Cubic B-spline kernel W(r, h).

    Parameters
    ----------
    q : ``|r| / h``  (dimensionless distance)
    h : smoothing length
    dim : spatial dimension (used for normalisation constant)

    Returns
    -------
    torch.Tensor  –  kernel values, same shape as q
    """
    if dim == 1:
        sigma = 2.0 / 3.0
    elif dim == 2:
        sigma = 10.0 / (7.0 * 3.14159265)
    else:
        sigma = 1.0 / 3.14159265

    alpha = sigma / (h ** dim)
    q = q.clamp(max=2.0)
    W = torch.where(
        q < 1.0,
        alpha * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3),
        torch.where(
            q < 2.0,
            alpha * 0.25 * (2.0 - q) ** 3,
            torch.zeros_like(q),
        ),
    )
    return W


def _cubic_kernel_grad(
    r_ij: torch.Tensor, dist: torch.Tensor, h: float, dim: int
) -> torch.Tensor:
    """Gradient of the cubic kernel: dW/dr_i.

    Parameters
    ----------
    r_ij  : ``(n_pairs, d)``  –  r_i - r_j
    dist  : ``(n_pairs,)``    –  |r_ij|
    h     : smoothing length
    dim   : spatial dimension

    Returns
    -------
    ``(n_pairs, d)``
    """
    if dim == 1:
        sigma = 2.0 / 3.0
    elif dim == 2:
        sigma = 10.0 / (7.0 * 3.14159265)
    else:
        sigma = 1.0 / 3.14159265

    alpha = sigma / (h ** dim)
    q = (dist / h).clamp(min=1e-10, max=2.0)
    dW_dq = torch.where(
        q < 1.0,
        alpha * (-3.0 * q + 2.25 * q ** 2),
        torch.where(q < 2.0, -alpha * 0.75 * (2.0 - q) ** 2, torch.zeros_like(q)),
    )  # (n_pairs,)
    # Chain rule: dW/dr = (dW/dq) * (dq/dr) = (dW/dq / (h * dist)) * r_ij
    dW_dr = dW_dq.unsqueeze(-1) / (h * dist.unsqueeze(-1).clamp(min=1e-12)) * r_ij
    return dW_dr  # (n_pairs, d)


# ---------------------------------------------------------------------------
# SPH Particles
# ---------------------------------------------------------------------------


class SPHParticles(ParticleSystem):
    """Weakly-Compressible Smoothed Particle Hydrodynamics (WCSPH).

    Models free-surface flows (water, droplets) using the standard WCSPH
    formulation.  Uses an explicit Euler integrator; replace with Verlet
    for improved energy conservation.

    Parameters
    ----------
    n_particles:
        Number of fluid particles.
    smoothing_length:
        SPH smoothing length *h* in world units.
    rho0:
        Reference density (kg m⁻³).
    c0:
        Reference speed of sound used in the Tait equation of state.
    nu:
        Kinematic viscosity (m² s⁻¹).
    dim:
        Spatial dimension (2 or 3).
    gamma:
        Exponent in Tait EOS (typically 7 for water).
    gravity:
        Gravitational acceleration vector of length *dim*.
    """

    def __init__(
        self,
        n_particles: int,
        smoothing_length: float,
        rho0: float = 1000.0,
        c0: float = 10.0,
        nu: float = 1e-4,
        dim: int = 2,
        gamma: float = 7.0,
        gravity: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__(n_particles, dim)
        self.h = smoothing_length
        self.rho0 = rho0
        self.c0 = c0
        self.nu = nu
        self.gamma = gamma

        if gravity is None:
            g = [0.0] * dim
            if dim >= 2:
                g[1] = -9.81
        else:
            g = list(gravity)[:dim]
        self.register_buffer("gravity", torch.tensor(g, dtype=torch.float32))

        # Background pressure coefficient B = rho0 * c0^2 / gamma
        self.B = rho0 * c0 ** 2 / gamma

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def density(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute particle densities via kernel summation.

        rho_i = sum_j m_j W(|r_i - r_j|, h)

        Assumes unit particle mass.

        Parameters
        ----------
        pos : ``(N, d)``

        Returns
        -------
        ``(N,)``  –  particle densities
        """
        r = pos.unsqueeze(0) - pos.unsqueeze(1)    # (N, N, d)
        dist = r.norm(dim=-1)                       # (N, N)
        q = dist / self.h
        W = _cubic_kernel(q, self.h, self.dim)      # (N, N)
        # rho_i = sum_j m_j W_ij  (m_j = 1 assumed)
        rho = W.sum(dim=1)                          # (N,)
        return rho.clamp(min=1e-4)

    def pressure(self, rho: torch.Tensor) -> torch.Tensor:
        """Tait equation of state: p = B * ((rho/rho0)^gamma - 1).

        Parameters
        ----------
        rho : ``(N,)``

        Returns
        -------
        ``(N,)``
        """
        return self.B * ((rho / self.rho0) ** self.gamma - 1.0)

    def _pressure_force(
        self,
        pos: torch.Tensor,
        rho: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric pressure force (anti-symmetric SPH gradient).

        F_press_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) * grad W_ij
        """
        i_idx, j_idx, r_ij = self._find_neighbours(pos, self.h)
        dist = r_ij.norm(dim=-1)
        dW = _cubic_kernel_grad(r_ij, dist, self.h, self.dim)

        p_term = (p[i_idx] / rho[i_idx] ** 2 + p[j_idx] / rho[j_idx] ** 2)
        f_ij = -p_term.unsqueeze(-1) * dW  # (n_pairs, d)

        F = torch.zeros_like(pos)
        F.scatter_add_(0, i_idx.unsqueeze(-1).expand(-1, self.dim), f_ij)
        return F

    def _viscous_force(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Laminar viscosity (artificial viscosity formulation).

        F_visc_i = nu * sum_j m_j (vel_j - vel_i) / rho_j * laplacian(W)_ij
        """
        i_idx, j_idx, r_ij = self._find_neighbours(pos, self.h)
        dist = r_ij.norm(dim=-1).clamp(min=1e-10)
        dW = _cubic_kernel_grad(r_ij, dist, self.h, self.dim)

        # Approximate Laplacian via Morris et al.
        dv = vel[j_idx] - vel[i_idx]          # (n_pairs, d)
        r_dot_dW = (r_ij * dW).sum(-1)        # (n_pairs,)
        r_sq = (r_ij ** 2).sum(-1).clamp(min=1e-10)
        lap_factor = 2.0 * r_dot_dW / (rho[j_idx] * r_sq)  # (n_pairs,)
        f_ij = self.nu * dv * lap_factor.unsqueeze(-1)

        F = torch.zeros_like(pos)
        F.scatter_add_(0, i_idx.unsqueeze(-1).expand(-1, self.dim), f_ij)
        return F

    # ------------------------------------------------------------------
    # Integration (explicit Euler)
    # ------------------------------------------------------------------

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the SPH system by one time step *dt*.

        Parameters
        ----------
        pos : ``(N, d)``  –  particle positions
        vel : ``(N, d)``  –  particle velocities
        dt  : time step

        Returns
        -------
        new_pos : ``(N, d)``
        new_vel : ``(N, d)``
        """
        rho = self.density(pos)                         # (N,)
        p = self.pressure(rho)                          # (N,)

        F_p = self._pressure_force(pos, rho, p)         # (N, d)
        F_v = self._viscous_force(pos, vel, rho)        # (N, d)

        g = self.gravity.to(pos.device)
        acc = (F_p + F_v) + g.unsqueeze(0)             # (N, d)

        new_vel = vel + dt * acc
        new_pos = pos + dt * new_vel
        return new_pos, new_vel
