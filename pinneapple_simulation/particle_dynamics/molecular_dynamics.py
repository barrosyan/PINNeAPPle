"""Molecular dynamics: pairwise Lennard-Jones potential with velocity-Verlet
time integration.

All operations are implemented in pure PyTorch so gradients flow through the
simulation for inverse problems and differentiable physics, matching the
house style of this package's other particle-based backends (SPH in
``particles.py``, MPM in ``mpm.py``, rigid bodies in ``rigid_body.py``).

Physics
-------
Lennard-Jones (LJ) pair potential -- a standard, well-documented, simple
interatomic potential for simple monatomic/noble-gas-like fluids (Jones,
1924; see e.g. Allen & Tildesley, "Computer Simulation of Liquids", 2nd ed.,
or Frenkel & Smit, "Understanding Molecular Simulation")::

    U(r) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

with the corresponding analytic pairwise force magnitude F(r) = -dU/dr,
directed along the line joining the two particles::

    F(r) = (24*epsilon/r) * (2*(sigma/r)**12 - (sigma/r)**6)

Time integration uses velocity-Verlet (Swope et al., J. Chem. Phys. 76, 637
(1982)) -- the standard symplectic integrator for microcanonical (NVE)
molecular dynamics::

    x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt**2
    v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

Boundary conditions use the minimum-image convention for periodic boundary
conditions (PBC): pairwise displacements are wrapped into [-L/2, L/2) before
computing forces/energies, and particle positions are wrapped back into
[0, L) after each step (Allen & Tildesley, op. cit., Sec. 1.5.2). Neither
``particles.py`` nor ``_utils.py`` in this package has an existing PBC
helper, so minimum-image PBC is implemented from scratch here.

This module implements a minimal NVE (constant particle-count/volume/energy)
simulator; thermostatting is intentionally out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class MDState:
    """Container for the state of an N-particle molecular dynamics system.

    Parameters
    ----------
    pos:
        ``(N, dim)`` -- particle positions.
    vel:
        ``(N, dim)`` -- particle velocities.
    """

    pos: torch.Tensor
    vel: torch.Tensor

    def clone(self) -> "MDState":
        """Return a deep copy of this state."""
        return MDState(pos=self.pos.clone(), vel=self.vel.clone())

    @property
    def n_particles(self) -> int:
        return self.pos.shape[0]

    @property
    def dim(self) -> int:
        return self.pos.shape[1]

    def __repr__(self) -> str:
        return f"MDState(n={self.n_particles}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Minimum-image-convention periodic boundary conditions
# ---------------------------------------------------------------------------


def _wrap_positions(pos: torch.Tensor, box_size: torch.Tensor) -> torch.Tensor:
    """Wrap positions back into ``[0, box_size)`` component-wise."""
    return pos - box_size * torch.floor(pos / box_size)


def _minimum_image(r_ij: torch.Tensor, box_size: torch.Tensor) -> torch.Tensor:
    """Apply the minimum-image convention to displacement vectors ``r_ij``.

    Assumes an orthorhombic periodic box of size ``box_size`` per axis and a
    cutoff <= box_size / 2 (the standard minimum-image-convention validity
    condition).
    """
    return r_ij - box_size * torch.round(r_ij / box_size)


# ---------------------------------------------------------------------------
# Lennard-Jones potential / force (standalone functions, unit-testable)
# ---------------------------------------------------------------------------


def lj_potential(dist: torch.Tensor, epsilon: float, sigma: float) -> torch.Tensor:
    """Lennard-Jones pair potential: U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)."""
    sr6 = (sigma / dist) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


def lj_force_magnitude(dist: torch.Tensor, epsilon: float, sigma: float) -> torch.Tensor:
    """Analytic radial Lennard-Jones force magnitude, F(r) = -dU/dr.

    F(r) = (24*epsilon/r) * (2*(sigma/r)^12 - (sigma/r)^6)

    A positive value is repulsive (pushes the pair apart along ``r_ij``).
    """
    sr6 = (sigma / dist) ** 6
    return (24.0 * epsilon / dist) * (2.0 * sr6 * sr6 - sr6)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MolecularDynamicsSimulator(nn.Module):
    """Minimal NVE molecular dynamics simulator.

    Pairwise Lennard-Jones interactions integrated with velocity-Verlet, with
    optional periodic boundary conditions (minimum-image convention).

    Uses brute-force O(N^2) neighbour search (as in ``ParticleSystem`` in
    ``particles.py``); replace with a cell list / neighbour list for large N.

    Parameters
    ----------
    epsilon:
        Lennard-Jones well depth.
    sigma:
        Lennard-Jones length scale (the distance at which U(r)=0).
    mass:
        Particle mass, assumed uniform across all particles.
    cutoff:
        Pairwise interaction cutoff radius. Defaults to ``2.5 * sigma``, the
        conventional LJ truncation distance. When periodic boundary
        conditions are used, ``cutoff`` must be <= half the box size for the
        minimum-image convention to remain valid.
    box_size:
        Optional periodic box size, either a scalar (cubic/square box, same
        size on every axis) or a ``(dim,)`` sequence. When given, periodic
        boundary conditions with the minimum-image convention are applied;
        when ``None`` the system has open (non-periodic) boundaries.
    dim:
        Spatial dimension (2 or 3). Only used to broadcast a scalar
        ``box_size`` to ``(dim,)``; otherwise purely informational.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        mass: float = 1.0,
        cutoff: Optional[float] = None,
        box_size: Optional[Tuple[float, ...]] = None,
        dim: int = 3,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.mass = float(mass)
        self.cutoff = float(cutoff) if cutoff is not None else 2.5 * self.sigma
        self.dim = dim

        if box_size is not None:
            bs = torch.as_tensor(box_size, dtype=torch.float32)
            if bs.ndim == 0:
                bs = bs.expand(dim).clone()
            self.register_buffer("box_size", bs)
        else:
            self.box_size = None

    # ------------------------------------------------------------------
    # Neighbour search
    # ------------------------------------------------------------------

    def _find_pairs(
        self, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(i_idx, j_idx, r_ij, dist)`` for ordered pairs within cutoff.

        ``r_ij[i] = pos[i_idx[i]] - pos[j_idx[i]]`` (minimum-image-wrapped
        when periodic).  Each unordered pair appears twice (as (i,j) and
        (j,i)), mirroring ``ParticleSystem._find_neighbours`` in
        ``particles.py``.
        """
        r = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, d), r[i, j] = pos_i - pos_j
        if self.box_size is not None:
            r = _minimum_image(r, self.box_size.to(pos.device))
        dist = r.norm(dim=-1)
        mask = (dist < self.cutoff) & (dist > 0.0)
        i_idx, j_idx = torch.where(mask)
        r_ij = r[i_idx, j_idx]
        d_ij = dist[i_idx, j_idx]
        return i_idx, j_idx, r_ij, d_ij

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def potential_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Total Lennard-Jones potential energy (each pair counted once)."""
        _, _, _, dist = self._find_pairs(pos)
        if dist.numel() == 0:
            return pos.new_zeros(())
        u = lj_potential(dist, self.epsilon, self.sigma)
        return 0.5 * u.sum()  # each unordered pair appears as (i,j) and (j,i)

    def kinetic_energy(self, vel: torch.Tensor) -> torch.Tensor:
        """Total kinetic energy: KE = 0.5 * m * sum(v^2)."""
        return 0.5 * self.mass * (vel ** 2).sum()

    def total_energy(self, state: MDState) -> torch.Tensor:
        """Total mechanical energy E = KE + PE (the NVE-conserved quantity)."""
        return self.potential_energy(state.pos) + self.kinetic_energy(state.vel)

    def compute_forces(self, pos: torch.Tensor) -> torch.Tensor:
        """Net Lennard-Jones force on each particle, ``(N, dim)``."""
        i_idx, j_idx, r_ij, dist = self._find_pairs(pos)
        F = torch.zeros_like(pos)
        if dist.numel() == 0:
            return F
        f_mag = lj_force_magnitude(dist, self.epsilon, self.sigma)  # (n_pairs,)
        f_vec = f_mag.unsqueeze(-1) * (r_ij / dist.unsqueeze(-1))   # force on i from j
        F.scatter_add_(0, i_idx.unsqueeze(-1).expand(-1, pos.shape[1]), f_vec)
        return F

    # ------------------------------------------------------------------
    # Integration (velocity-Verlet)
    # ------------------------------------------------------------------

    def step(self, state: MDState, dt: float) -> MDState:
        """Advance the system by one velocity-Verlet step of size *dt*.

        Velocity-Verlet (Swope et al. 1982) is the standard symplectic
        integrator for NVE molecular dynamics::

            x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            v(t+dt) = v(t) + 0.5*(a(t)+a(t+dt))*dt
        """
        F0 = self.compute_forces(state.pos)
        a0 = F0 / self.mass

        new_pos = state.pos + state.vel * dt + 0.5 * a0 * dt * dt
        if self.box_size is not None:
            new_pos = _wrap_positions(new_pos, self.box_size.to(new_pos.device))

        F1 = self.compute_forces(new_pos)
        a1 = F1 / self.mass
        new_vel = state.vel + 0.5 * (a0 + a1) * dt

        return MDState(pos=new_pos, vel=new_vel)

    def forward(self, state: MDState, n_steps: int = 1, dt: float = 1e-3) -> MDState:
        """Advance the system by ``n_steps`` velocity-Verlet steps of size ``dt``."""
        for _ in range(n_steps):
            state = self.step(state, dt)
        return state
