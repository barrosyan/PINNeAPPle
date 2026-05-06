"""Boundary handling helpers for SPH (MVP).

This module provides simple boundary particle generation and wall forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BoxBoundary:
    """Axis-aligned box boundary [min,max] in each dimension."""

    lo: torch.Tensor  # (D,)
    hi: torch.Tensor  # (D,)

    @staticmethod
    def from_bounds(lo, hi, device=None, dtype=torch.float32) -> "BoxBoundary":
        return BoxBoundary(
            lo=torch.tensor(lo, device=device, dtype=dtype),
            hi=torch.tensor(hi, device=device, dtype=dtype),
        )


def wall_repulsion_force(pos: torch.Tensor, vel: torch.Tensor, boundary: BoxBoundary, *, k: float = 50.0, d0: float = 0.02) -> torch.Tensor:
    """Soft wall repulsion: pushes particles back when closer than d0."""
    D = pos.shape[1]
    f = torch.zeros_like(pos)
    for d in range(D):
        dist_lo = pos[:, d] - boundary.lo[d]
        dist_hi = boundary.hi[d] - pos[:, d]
        # lo
        mask = dist_lo < d0
        f[mask, d] += k * (d0 - dist_lo[mask])
        # hi
        mask = dist_hi < d0
        f[mask, d] -= k * (d0 - dist_hi[mask])
    return f


def reflect_box(
    pos: torch.Tensor,
    vel: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    *,
    restitution: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hard boundary reflection for an axis-aligned box.

    Parameters
    ----------
    pos : (N, D)
    vel : (N, D)
    lo  : (D,)
    hi  : (D,)
    restitution : float
        0.0 = inelastic (stick), 1.0 = perfectly elastic

    Returns
    -------
    pos_reflected, vel_reflected
    """
    pos = pos.clone()
    vel = vel.clone()

    N, D = pos.shape

    for d in range(D):
        # --- lower bound ---
        mask_lo = pos[:, d] < lo[d]
        if mask_lo.any():
            pos[mask_lo, d] = lo[d]
            vel[mask_lo, d] = -restitution * vel[mask_lo, d]

        # --- upper bound ---
        mask_hi = pos[:, d] > hi[d]
        if mask_hi.any():
            pos[mask_hi, d] = hi[d]
            vel[mask_hi, d] = -restitution * vel[mask_hi, d]

    return pos, vel
