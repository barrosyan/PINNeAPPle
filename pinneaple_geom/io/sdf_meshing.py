
"""Meshing utilities for implicit SDF geometries.

Provides a practical path:
  SDF(params) -> boundary sampling (marching squares, 2D) -> triangle mesh via gmsh (optional)

This is meant for *design iteration* loops where geometry parameters change frequently.

Dependencies:
  - numpy (required)
  - scikit-image (optional) for marching squares; fallback to a simple implementation if missing
  - gmsh + meshio (optional) to triangulate contours into MeshData

If gmsh is not available, you can still use `sample_boundary_points_sdf2d`
to get boundary point clouds for PINNs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.io.meshio_bridge import _require_meshio, load_meshio


SDF2D = Callable[[np.ndarray], np.ndarray]
# points input: (N,2) in physical space; output: (N,) signed distance (negative inside)


@dataclass
class SDFGrid2D:
    bounds_min: Tuple[float, float] = (-1.0, -1.0)
    bounds_max: Tuple[float, float] = (1.0, 1.0)
    resolution: int = 256


def _grid_points(cfg: SDFGrid2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(cfg.bounds_min[0], cfg.bounds_max[0], cfg.resolution)
    ys = np.linspace(cfg.bounds_min[1], cfg.bounds_max[1], cfg.resolution)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    return xs, ys, pts


def sample_boundary_points_sdf2d(
    sdf: SDF2D,
    *,
    grid: Optional[SDFGrid2D] = None,
    n: int = 2048,
    rng: Optional[np.random.Generator] = None,
    band: float = 0.01,
) -> np.ndarray:
    """Sample boundary points from a 2D SDF by rejection within a thin band."""
    rng = rng or np.random.default_rng()
    grid = grid or SDFGrid2D()
    xs = rng.uniform(grid.bounds_min[0], grid.bounds_max[0], size=(max(n * 50, 10000),))
    ys = rng.uniform(grid.bounds_min[1], grid.bounds_max[1], size=(max(n * 50, 10000),))
    pts = np.stack([xs, ys], axis=-1)
    d = sdf(pts)
    mask = np.abs(d) <= float(band)
    cand = pts[mask]
    if cand.shape[0] < n:
        # fallback: take nearest to boundary
        idx = np.argsort(np.abs(d))[:n]
        return pts[idx].astype(np.float64)
    idx = rng.choice(cand.shape[0], size=n, replace=False)
    return cand[idx].astype(np.float64)


def marching_squares_contours(
    sdf: SDF2D,
    *,
    grid: Optional[SDFGrid2D] = None,
    level: float = 0.0,
) -> list[np.ndarray]:
    """Extract contour polylines for the level set sdf(x)=level.

    Returns a list of polylines, each shaped (M,2).

    Uses scikit-image if available.
    """
    grid = grid or SDFGrid2D()
    xs, ys, pts = _grid_points(grid)
    vals = sdf(pts).reshape(grid.resolution, grid.resolution)

    try:
        from skimage import measure  # type: ignore
        cs = measure.find_contours(vals, level=level)
        polys = []
        for c in cs:
            # c is (K,2) in (row,col) index coords; map to physical x/y
            rr = c[:, 0]
            cc = c[:, 1]
            x = np.interp(cc, np.arange(grid.resolution), xs)
            y = np.interp(rr, np.arange(grid.resolution), ys)
            polys.append(np.stack([x, y], axis=-1))
        return polys
    except Exception:
        # minimal fallback: no contour extraction
        raise ImportError("scikit-image is required for marching_squares_contours. Install: pip install scikit-image")


def _require_gmsh():
    try:
        import gmsh  # type: ignore
    except Exception as e:
        raise ImportError("gmsh is required for triangulating SDF contours. Install: pip install gmsh") from e
    return gmsh


def sdf2d_to_tri_mesh(
    sdf: SDF2D,
    *,
    grid: Optional[SDFGrid2D] = None,
    mesh_size: float = 0.03,
    cache_path: Optional[str] = None,
) -> MeshData:
    """Build a 2D triangulated mesh of the zero level-set using gmsh.

    Produces a *surface* mesh embedded in 2D (z=0) as triangles.

    If you only need boundary points for PINNs, prefer `sample_boundary_points_sdf2d`.
    """
    _require_meshio()
    gmsh = _require_gmsh()
    grid = grid or SDFGrid2D()
    polys = marching_squares_contours(sdf, grid=grid, level=0.0)
    if len(polys) == 0:
        raise RuntimeError("No contours found for SDF. Check bounds/resolution.")

    # pick largest contour by perimeter
    def perim(p):
        d = np.linalg.norm(p[1:] - p[:-1], axis=-1).sum()
        d += np.linalg.norm(p[0] - p[-1])
        return d
    poly = max(polys, key=perim)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("pinneaple_sdf2d")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(mesh_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(mesh_size))

        # add points
        pts_tags = []
        for i, (x, y) in enumerate(poly):
            pts_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(mesh_size)))

        # close loop
        lines = []
        for i in range(len(pts_tags)):
            a = pts_tags[i]
            b = pts_tags[(i + 1) % len(pts_tags)]
            lines.append(gmsh.model.geo.addLine(a, b))
        cloop = gmsh.model.geo.addCurveLoop(lines)
        surf = gmsh.model.geo.addPlaneSurface([cloop])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        out_msh = cache_path or "/tmp/pinneaple_sdf2d.msh"
        gmsh.write(out_msh)
    finally:
        gmsh.finalize()

    import meshio  # type: ignore
    msh = meshio.read(out_msh)
    return load_meshio(msh)
