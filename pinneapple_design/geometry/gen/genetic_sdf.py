
from __future__ import annotations

"""Genetic (parametric) implicit geometry via SDF.

This module provides a lightweight, dependency-free way to define *real* "genetic"
geometries for Arena/PINNs:
  - A genome is a dict of parameters.
  - The shape is represented by a Signed Distance Function (SDF).
  - You can sample inside/outside points and extract the boundary via marching squares.

Current primitive:
  - Union of circles, optionally intersected with a bounding box.

You can extend this to capsules, superquadrics, splines, etc.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


def _sdf_box(p: torch.Tensor, bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    # SDF for axis-aligned box. Negative inside.
    # Based on standard IQ formulation.
    c = 0.5 * (bmin + bmax)
    h = 0.5 * (bmax - bmin)
    q = (p - c).abs() - h
    # outside distance
    out = torch.clamp(q, min=0.0).norm(dim=-1)
    # inside distance (negative)
    inn = torch.clamp(q.max(dim=-1).values, max=0.0)
    return out + inn


def sdf_union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b)


def sdf_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


@dataclass
class GeneticSDF:
    """A callable SDF domain with an explicit genome."""

    sdf: Callable[[torch.Tensor], torch.Tensor]
    genome: Dict[str, float]
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    dim: int = 2

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return self.sdf(p)


def make_union_circles_sdf(
    *,
    n_circles: int = 3,
    seed: int = 0,
    bounds_min: Tuple[float, float] = (0.0, 0.0),
    bounds_max: Tuple[float, float] = (1.0, 1.0),
    r_min: float = 0.12,
    r_max: float = 0.28,
    intersect_box: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> GeneticSDF:
    """Create a random "genetic" domain as union of circles.

    Returns:
      GeneticSDF where sdf(p) < 0 means inside domain.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    bmin = torch.tensor(bounds_min, device=device, dtype=dtype)
    bmax = torch.tensor(bounds_max, device=device, dtype=dtype)

    # random centers (avoid edges)
    margin = 0.15
    centers = margin + (1 - 2 * margin) * torch.rand((n_circles, 2), generator=g, dtype=dtype)
    centers = bmin + centers * (bmax - bmin)

    radii = r_min + (r_max - r_min) * torch.rand((n_circles,), generator=g, dtype=dtype)

    genome: Dict[str, float] = {}
    for i in range(n_circles):
        genome[f"c{i}_x"] = float(centers[i, 0].item())
        genome[f"c{i}_y"] = float(centers[i, 1].item())
        genome[f"r{i}"] = float(radii[i].item())

    def sdf(p: torch.Tensor) -> torch.Tensor:
        # p: (...,2)
        d = None
        for i in range(n_circles):
            ci = torch.tensor([genome[f"c{i}_x"], genome[f"c{i}_y"]], device=p.device, dtype=p.dtype)
            ri = torch.tensor(genome[f"r{i}"], device=p.device, dtype=p.dtype)
            di = (p - ci).norm(dim=-1) - ri
            d = di if d is None else sdf_union(d, di)

        if intersect_box:
            db = _sdf_box(p, bmin.to(p.device, p.dtype), bmax.to(p.device, p.dtype))
            d = sdf_intersection(d, db)
        return d

    return GeneticSDF(sdf=sdf, genome=genome, bounds_min=bmin, bounds_max=bmax, dim=2)
