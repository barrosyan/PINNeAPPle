"""Uniform and Latin hypercube sampling in axis-aligned boxes."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def sample_uniform_box(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Uniform random samples inside an axis-aligned bounding box.

    bounds_min/max: (3,)
    returns: (n,3)
    """
    rng = rng or np.random.default_rng()
    bounds_min = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    bounds_max = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    u = rng.random((n, 3))
    return bounds_min[None, :] + u * (bounds_max - bounds_min)[None, :]


def sample_latin_hypercube_box(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS) in an axis-aligned box.
    Good for collocation coverage without clustering.

    bounds_min/max: (d,) (commonly d=3)
    returns: (n,d)
    """
    rng = rng or np.random.default_rng()
    bounds_min = np.asarray(bounds_min, dtype=np.float64).reshape(-1)
    bounds_max = np.asarray(bounds_max, dtype=np.float64).reshape(-1)
    d = bounds_min.shape[0]

    # LHS in [0,1]^d
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1:]
    rdpoints = u * (b - a)[:, None] + a[:, None]  # (n,1) broadcast to (n,d)

    # permute per dimension
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, 0]

    # scale to bounds
    return bounds_min[None, :] + H * (bounds_max - bounds_min)[None, :]
