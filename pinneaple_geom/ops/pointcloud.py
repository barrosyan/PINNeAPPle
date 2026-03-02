
"""Point cloud utilities.

Conventions:
  - point cloud: torch.Tensor (N,3) or (N,2)
  - optional normals: torch.Tensor (N,3)
  - optional features: torch.Tensor (N,F)

This module bridges geometry (mesh/SDF) -> point clouds usable by:
  - PINNs (collocation/boundary points)
  - Neural operators on points (e.g., UNO points mode, GNO)
  - Graph-based models (GNNs) after building edges / kNN
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch

from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.sample.points import sample_surface_points
from pinneaple_geom.io.sdf_meshing import sample_boundary_points_sdf2d, SDF2D, SDFGrid2D


@dataclass
class PointCloud:
    points: torch.Tensor          # (N,D)
    normals: Optional[torch.Tensor] = None  # (N,3)
    features: Optional[torch.Tensor] = None # (N,F)


def mesh_to_pointcloud(
    mesh: MeshData,
    *,
    n_surface: int = 4096,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    include_normals: bool = True,
) -> PointCloud:
    rng = np.random.default_rng(int(seed))
    pts, normals, _ = sample_surface_points(mesh, n_surface, rng=rng, return_normals=include_normals, return_face_id=False)
    p = torch.tensor(pts, device=device, dtype=dtype)
    n = torch.tensor(normals, device=device, dtype=dtype) if (include_normals and normals is not None) else None
    return PointCloud(points=p, normals=n, features=None)


def sample_pointcloud_from_mesh(points: np.ndarray, cells, *, n: int = 8192, seed: int = 0) -> np.ndarray:
    """Convenience wrapper used by the webapp.

    Parameters
    - points: (N,3) numpy
    - cells: list[CellBlock] (as in MeshData.cells)
    - n: number of surface samples

    Returns
    - pc: (n,3) numpy array
    """
    mesh = MeshData(points=points, cells=cells)
    pc = mesh_to_pointcloud(mesh, n_surface=int(n), seed=int(seed), device=None, dtype=torch.float32, include_normals=False)
    return pc.points.detach().cpu().numpy()


def sdf2d_to_pointcloud(
    sdf: SDF2D,
    *,
    grid: Optional[SDFGrid2D] = None,
    n_boundary: int = 4096,
    band: float = 0.01,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PointCloud:
    rng = np.random.default_rng(int(seed))
    pts = sample_boundary_points_sdf2d(sdf, grid=grid, n=n_boundary, rng=rng, band=band)
    p = torch.tensor(pts, device=device, dtype=dtype)
    return PointCloud(points=p, normals=None, features=None)
