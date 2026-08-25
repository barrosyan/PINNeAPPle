"""
Point-cloud dataset augmentation strategies for PDE/PINN training data.

Operates on the same plain `(coords, fields)` representation used throughout
this package (`coords`: (N, d) array; `fields`: {name: (N,) or (N, k) array}).
No framework/storage coupling — callers own loading and persisting datasets.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _sample_inside_hull(coords: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `n_target` new points uniformly inside the convex hull of
    `coords` (falls back to bounding-box sampling for degenerate/1D point
    clouds, or if the hull construction fails)."""
    ndim = coords.shape[1]
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    if ndim == 1:
        return rng.uniform(lo, hi, size=(n_target, 1))
    from scipy.spatial import Delaunay, QhullError

    try:
        tri = Delaunay(coords)
    except QhullError:
        return rng.uniform(lo, hi, size=(n_target, ndim))
    pts: list = []
    attempts = 0
    while len(pts) < n_target and attempts < 50:
        batch = max((n_target - len(pts)) * 4, 100)
        cand = rng.uniform(lo, hi, size=(batch, ndim))
        inside = cand[tri.find_simplex(cand) >= 0]
        pts.extend(list(inside))
        attempts += 1
    if not pts:
        return rng.uniform(lo, hi, size=(n_target, ndim))
    pts = np.asarray(pts[:n_target])
    if len(pts) < n_target:  # pad with bbox samples if the hull search fell short
        pad = rng.uniform(lo, hi, size=(n_target - len(pts), ndim))
        pts = np.concatenate([pts, pad], axis=0)
    return pts


def augment_noise(
    coords: np.ndarray,
    fields: Dict[str, np.ndarray],
    n_copies: int,
    std_fraction: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Add Gaussian noise (scaled to each field's own std) to `n_copies`
    tiled copies of the dataset. Coordinates are unchanged; only field values
    are perturbed."""
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float32)
    new_coords = np.tile(coords, (n_copies, 1))
    new_fields = {}
    for k, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float32)
        std = float(arr.std()) * std_fraction
        if std <= 0:
            std = 1e-6
        new_fields[k] = np.concatenate(
            [arr + rng.normal(0.0, std, size=arr.shape).astype(np.float32) for _ in range(n_copies)]
        )
    return {"coords": new_coords, "fields": new_fields,
            "info": {"strategy": "noise", "std_fraction": std_fraction, "n_copies": n_copies}}


def augment_transform(
    coords: np.ndarray,
    fields: Dict[str, np.ndarray],
    n_copies: int,
    max_rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    seed: int = 0,
) -> Dict[str, Any]:
    """Random in-plane rotation (about the first two coordinate axes) plus
    isotropic scaling of the coordinate frame, `n_copies` times. Field values
    are treated as physical quantities sampled at each point, not
    coordinate-derived geometry — a rigid rotation/scale of the frame leaves
    field *values* unchanged, so they are simply tiled to match the
    (repeated) coordinate count."""
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float32)
    ndim = coords.shape[1]
    chunks = []
    for _ in range(n_copies):
        angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
        scale = rng.uniform(*scale_range)
        c = coords.copy()
        if ndim >= 2:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x, y = c[:, 0].copy(), c[:, 1].copy()
            c[:, 0] = (cos_a * x - sin_a * y) * scale
            c[:, 1] = (sin_a * x + cos_a * y) * scale
            if ndim > 2:
                c[:, 2:] = c[:, 2:] * scale
        else:
            c = c * scale
        chunks.append(c)
    new_coords = np.concatenate(chunks, axis=0)
    new_fields = {k: np.tile(np.asarray(v, dtype=np.float32), n_copies) for k, v in fields.items()}
    return {"coords": new_coords, "fields": new_fields,
            "info": {"strategy": "transform", "max_rotation_deg": max_rotation_deg,
                      "scale_range": list(scale_range)}}


def augment_interpolation(
    coords: np.ndarray,
    fields: Dict[str, np.ndarray],
    n_new_points: int,
    method: str = "linear",
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate `n_new_points` new samples inside the convex hull of the
    original point cloud, with field values estimated via scattered-data
    interpolation (falling back to nearest-neighbor wherever the requested
    method produces NaN, e.g. outside the interpolation method's support)."""
    from scipy.interpolate import griddata

    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float32)
    new_coords = _sample_inside_hull(coords, n_new_points, rng).astype(np.float32)
    new_fields = {}
    for k, arr in fields.items():
        arr = np.asarray(arr, dtype=np.float32)
        vals = griddata(coords, arr, new_coords, method=method)
        vals = np.asarray(vals, dtype=np.float32)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            fallback = griddata(coords, arr, new_coords[nan_mask], method="nearest")
            vals[nan_mask] = np.asarray(fallback, dtype=np.float32)
        new_fields[k] = vals
    return {"coords": new_coords, "fields": new_fields,
            "info": {"strategy": "interpolation", "target_points": n_new_points, "method": method}}
