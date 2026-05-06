"""Barycentric sampling and interpolation on triangles."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def sample_barycentric_uv(n: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample barycentric weights (w0, w1, w2) uniformly over a triangle.

    Uses the classic trick:
      u, v ~ U(0,1)
      su = sqrt(u)
      w0 = 1 - su
      w1 = su * (1 - v)
      w2 = su * v
    """
    rng = rng or np.random.default_rng()
    u = rng.random(n)
    v = rng.random(n)
    su = np.sqrt(u)

    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    return w0, w1, w2


def sample_points_on_triangles(
    tri_v0: np.ndarray,
    tri_v1: np.ndarray,
    tri_v2: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample n points uniformly on a single triangle defined by (v0,v1,v2).

    tri_v*: shape (3,)
    returns: (n,3)
    """
    rng = rng or np.random.default_rng()
    w0, w1, w2 = sample_barycentric_uv(n, rng=rng)
    pts = (w0[:, None] * tri_v0[None, :]) + (w1[:, None] * tri_v1[None, :]) + (w2[:, None] * tri_v2[None, :])
    return pts.astype(np.float64)


def interpolate_on_triangles(
    values_v0: np.ndarray,
    values_v1: np.ndarray,
    values_v2: np.ndarray,
    w0: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    """
    Barycentric interpolation of any per-vertex values.

    values_v*: shape (n, d) OR (d,) broadcastable
    w*: shape (n,)
    returns: shape (n, d)
    """
    return (w0[:, None] * values_v0) + (w1[:, None] * values_v1) + (w2[:, None] * values_v2)
