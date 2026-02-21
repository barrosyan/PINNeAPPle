"""Surface and boundary point sampling for triangle meshes."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.ops.features import compute_face_normals, compute_face_areas
from .barycentric import sample_barycentric_uv


def _choice_weighted(rng: np.random.Generator, weights: np.ndarray, n: int) -> np.ndarray:
    """
    Weighted random choice over faces.
    weights: (M,) non-negative
    returns: (n,) indices
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be > 0.")
    p = w / s
    return rng.choice(len(w), size=n, replace=True, p=p)


def sample_surface_points(
    mesh: MeshData,
    n: int,
    *,
    rng: Optional[np.random.Generator] = None,
    return_normals: bool = True,
    return_face_id: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample n points uniformly over the surface (triangle areas as weights).

    Returns:
      points: (n,3)
      normals: (n,3) or None
      face_id: (n,) or None
    """
    rng = rng or np.random.default_rng()

    areas = compute_face_areas(mesh)
    face_id = _choice_weighted(rng, areas, n)

    v = mesh.vertices
    f = mesh.faces

    v0 = v[f[face_id, 0]]
    v1 = v[f[face_id, 1]]
    v2 = v[f[face_id, 2]]

    w0, w1, w2 = sample_barycentric_uv(n, rng=rng)
    pts = (w0[:, None] * v0) + (w1[:, None] * v1) + (w2[:, None] * v2)

    normals = None
    if return_normals:
        fn = mesh.normals
        if fn is None or fn.shape[0] != mesh.faces.shape[0]:
            fn = compute_face_normals(mesh)
        normals = fn[face_id]

    out_face = face_id.astype(np.int64) if return_face_id else None
    return pts.astype(np.float64), (None if normals is None else normals.astype(np.float64)), out_face


def sample_surface_points_weighted(
    mesh: MeshData,
    n: int,
    *,
    face_weights: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    return_normals: bool = True,
    return_face_id: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample n points on surface using user-provided per-face weights.

    Useful for importance sampling (e.g., high curvature regions).
    """
    rng = rng or np.random.default_rng()

    face_id = _choice_weighted(rng, face_weights, n)

    v = mesh.vertices
    f = mesh.faces

    v0 = v[f[face_id, 0]]
    v1 = v[f[face_id, 1]]
    v2 = v[f[face_id, 2]]

    w0, w1, w2 = sample_barycentric_uv(n, rng=rng)
    pts = (w0[:, None] * v0) + (w1[:, None] * v1) + (w2[:, None] * v2)

    normals = None
    if return_normals:
        fn = mesh.normals
        if fn is None or fn.shape[0] != mesh.faces.shape[0]:
            fn = compute_face_normals(mesh)
        normals = fn[face_id]

    out_face = face_id.astype(np.int64) if return_face_id else None
    return pts.astype(np.float64), (None if normals is None else normals.astype(np.float64)), out_face
