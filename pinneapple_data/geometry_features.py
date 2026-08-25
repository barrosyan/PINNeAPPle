"""
Geometry featurization: surface point-cloud sampling and a coarse
signed-distance-field (SDF) grid from a triangle mesh — generic inputs
(vertices/faces arrays, matching this package's own `stl_import.STLMesh`
convention), no file-format or storage coupling. Useful as a preprocessing
step for any geometry-conditioned model (e.g. a GNN or point-cloud PINN that
needs an explicit distance-to-surface feature).

Requires `trimesh` (only used internally for surface sampling and the
signed-distance query — both nontrivial geometry algorithms not worth
re-implementing).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def sample_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int = 2048,
) -> np.ndarray:
    """Sample `n_samples` points uniformly (area-weighted) from a mesh's
    surface. `n_samples` is silently capped at 3x the face count for a very
    coarse mesh, since sampling far more points than the mesh has resolution
    to support doesn't add real information."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    n = min(int(n_samples), max(len(mesh.faces), 1) * 3)
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(points)


def compute_sdf_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_resolution: int = 16,
    bounds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate a signed distance field on a regular grid spanning the mesh's
    bounding box (or a caller-supplied `bounds` = [[xmin,ymin,zmin],
    [xmax,ymax,zmax]], e.g. to featurize several meshes on one shared grid).

    Returns {"points": (grid_resolution^3, 3), "sdf": (grid_resolution^3,),
    "grid_shape": [grid_resolution]*3}. Positive values are outside the mesh,
    negative inside (trimesh's convention).
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    b = np.asarray(bounds) if bounds is not None else mesh.bounds
    if b is None:
        raise ValueError("mesh has no valid bounds and none were supplied")

    lin = [np.linspace(b[0][i], b[1][i], grid_resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(*lin, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    sdf = trimesh.proximity.signed_distance(mesh, grid_pts)
    return {"points": grid_pts, "sdf": np.asarray(sdf), "grid_shape": [grid_resolution] * 3}


def featurize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_point_cloud_samples: int = 2048,
    sdf_grid_resolution: int = 16,
) -> Dict[str, Any]:
    """Convenience wrapper: point cloud + SDF grid in one call, with
    per-stage errors reported rather than raised (a degenerate/non-watertight
    mesh may still yield a usable point cloud even if the SDF query fails)."""
    result: Dict[str, Any] = {}
    try:
        result["point_cloud"] = sample_point_cloud(vertices, faces, n_point_cloud_samples)
    except Exception as exc:
        result["point_cloud_error"] = str(exc)

    try:
        result["sdf"] = compute_sdf_grid(vertices, faces, sdf_grid_resolution)
    except Exception as exc:
        result["sdf_error"] = str(exc)

    return result
