
from __future__ import annotations

"""Boundary extraction from SDF via marching squares (matplotlib contour).

This is a practical MVP:
  - Evaluate SDF on a grid
  - Use matplotlib's contour (marching squares) to extract the 0-level set
  - Sample points uniformly from the polyline vertices

No extra dependencies beyond matplotlib/numpy.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import torch


def marching_squares_boundary(
    sdf: Callable[[torch.Tensor], torch.Tensor],
    *,
    bounds_min: Tuple[float, float] = (0.0, 0.0),
    bounds_max: Tuple[float, float] = (1.0, 1.0),
    resolution: int = 256,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return boundary polyline vertices as (M,2) tensor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bmin = torch.tensor(bounds_min, device=device, dtype=dtype)
    bmax = torch.tensor(bounds_max, device=device, dtype=dtype)

    xs = torch.linspace(bmin[0], bmax[0], resolution, device=device, dtype=dtype)
    ys = torch.linspace(bmin[1], bmax[1], resolution, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    P = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        Z = sdf(P).reshape(resolution, resolution).detach().cpu().numpy()

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot(111)
    cs = ax.contour(xs.cpu().numpy(), ys.cpu().numpy(), Z, levels=[0.0])
    plt.close(fig)

    verts = []
    for col in cs.collections:
        for p in col.get_paths():
            v = p.vertices
            if v.shape[0] >= 2:
                verts.append(v)
    if not verts:
        return torch.zeros((0, 2), device=device, dtype=dtype)

    V = np.concatenate(verts, axis=0)
    return torch.tensor(V, device=device, dtype=dtype)


def sample_boundary_points(
    sdf: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    *,
    bounds_min: Tuple[float, float] = (0.0, 0.0),
    bounds_max: Tuple[float, float] = (1.0, 1.0),
    resolution: int = 256,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Sample n boundary points from the extracted contour vertices."""
    V = marching_squares_boundary(
        sdf,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        resolution=resolution,
        device=device,
        dtype=dtype,
    )
    if V.numel() == 0:
        return V
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx = torch.randint(0, V.shape[0], (int(n),), generator=g)
    return V[idx]
