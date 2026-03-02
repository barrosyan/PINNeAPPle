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
