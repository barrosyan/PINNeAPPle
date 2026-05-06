"""Particle samplers for SPH and particle-based datasets."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def sample_box_particles(
    *,
    lo: Tuple[float, ...],
    hi: Tuple[float, ...],
    spacing: float,
    jitter: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Regular grid of particles in an axis-aligned box."""
    lo_t = torch.tensor(lo, device=device, dtype=dtype)
    hi_t = torch.tensor(hi, device=device, dtype=dtype)
    D = lo_t.numel()
    spacing = float(spacing)

    grids = []
    for d in range(D):
        n = int(torch.floor((hi_t[d] - lo_t[d]) / spacing).item()) + 1
        grids.append(torch.linspace(lo_t[d], lo_t[d] + spacing * (n - 1), n, device=device, dtype=dtype))
    mesh = torch.meshgrid(*grids, indexing="xy" if D == 2 else "ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], dim=1)
    if jitter > 0:
        pts = pts + (torch.rand_like(pts) - 0.5) * float(jitter) * spacing
    return pts


def sample_circle_particles(
    *,
    center: Tuple[float, float],
    radius: float,
    spacing: float,
    jitter: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2D particles inside a circle."""
    cx, cy = center
    r = float(radius)
    pts = sample_box_particles(lo=(cx - r, cy - r), hi=(cx + r, cy + r), spacing=spacing, jitter=jitter, device=device, dtype=dtype)
    d = torch.norm(pts - torch.tensor([cx, cy], device=device, dtype=dtype), dim=1)
    return pts[d <= r]
