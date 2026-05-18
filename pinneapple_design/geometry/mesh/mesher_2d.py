"""2D mesh generation utilities.

Provides multiple backends for triangulating 2D domains:
- Structured rectangular grids (no external dependencies)
- Quality triangle meshes from SDF via gmsh (optional)
- Constrained Delaunay triangulation from boundary polygons (scipy, optional)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Mesh2D:
    """Lightweight 2D triangle mesh.

    Attributes
    ----------
    vertices  : (N, 2) float32 node positions
    triangles : (M, 3) int32 triangle connectivity
    boundary_edges : dict mapping region name -> (K, 2) vertex index pairs
    boundary_points : dict mapping region name -> vertex indices on that boundary
    meta : free-form metadata
    """
    vertices: np.ndarray        # (N, 2)
    triangles: np.ndarray       # (M, 3)
    boundary_edges: Dict[str, np.ndarray] = field(default_factory=dict)
    boundary_points: Dict[str, List[int]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def triangle_centroids(self) -> np.ndarray:
        """(M, 2) centroid of each triangle."""
        v = self.vertices
        t = self.triangles
        return (v[t[:, 0]] + v[t[:, 1]] + v[t[:, 2]]) / 3.0

    def triangle_areas(self) -> np.ndarray:
        """(M,) area of each triangle."""
        v = self.vertices
        t = self.triangles
        a = v[t[:, 1]] - v[t[:, 0]]
        b = v[t[:, 2]] - v[t[:, 0]]
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def min_angles(self) -> np.ndarray:
        """(M,) minimum interior angle per triangle (degrees)."""
        v = self.vertices
        t = self.triangles
        angles = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            a = v[t[:, j]] - v[t[:, i]]
            b = v[t[:, k]] - v[t[:, i]]
            cos_a = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        return np.min(np.stack(angles, axis=1), axis=1)

    def quality_stats(self) -> Dict[str, float]:
        """Summary mesh quality statistics."""
        min_angs = self.min_angles()
        areas = self.triangle_areas()
        return {
            "n_vertices": self.n_vertices,
            "n_triangles": self.n_triangles,
            "min_angle_mean": float(np.mean(min_angs)),
            "min_angle_min": float(np.min(min_angs)),
            "area_mean": float(np.mean(areas)),
            "area_std": float(np.std(areas)),
            "total_area": float(np.sum(areas)),
        }

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        """Sample n points inside the mesh using barycentric sampling."""
        rng = np.random.default_rng(seed)
        areas = self.triangle_areas()
        total = areas.sum()
        if total <= 0:
            return np.zeros((n, 2), dtype=np.float32)
        probs = areas / total
        tri_idx = rng.choice(len(self.triangles), size=n, p=probs)

        v = self.vertices
        t = self.triangles[tri_idx]
        r1 = rng.random(n).astype(np.float32)
        r2 = rng.random(n).astype(np.float32)
        # barycentric
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v_b = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2
        pts = (u[:, None] * v[t[:, 0]] + v_b[:, None] * v[t[:, 1]] + w[:, None] * v[t[:, 2]])
        return pts.astype(np.float32)


# ---------------------------------------------------------------------------
# Structured rectangular grid
# ---------------------------------------------------------------------------

def mesh_rectangle_structured(
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    nx: int = 20,
    ny: int = 20,
) -> Mesh2D:
    """Generate a structured triangular mesh of a rectangle.

    Each grid cell is split into 2 triangles.

    Parameters
    ----------
    x_range, y_range : domain extents
    nx, ny : number of cells in each direction

    Returns
    -------
    Mesh2D
    """
    x = np.linspace(x_range[0], x_range[1], nx + 1, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], ny + 1, dtype=np.float32)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    verts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # ((nx+1)*(ny+1), 2)

    def vid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            v0, v1, v2, v3 = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])

    triangles = np.array(tris, dtype=np.int32)

    # Boundary groups
    def _collect(mask):
        return np.where(mask)[0].tolist()

    bmin_x, bmax_x = x_range
    bmin_y, bmax_y = y_range
    tol = 1e-6 * max(bmax_x - bmin_x, bmax_y - bmin_y)

    bp = {
        "inlet":  _collect(np.abs(verts[:, 0] - bmin_x) < tol),
        "outlet": _collect(np.abs(verts[:, 0] - bmax_x) < tol),
        "bottom": _collect(np.abs(verts[:, 1] - bmin_y) < tol),
        "top":    _collect(np.abs(verts[:, 1] - bmax_y) < tol),
        "walls":  _collect(
            (np.abs(verts[:, 1] - bmin_y) < tol) | (np.abs(verts[:, 1] - bmax_y) < tol)
        ),
        "boundary": _collect(
            (np.abs(verts[:, 0] - bmin_x) < tol) | (np.abs(verts[:, 0] - bmax_x) < tol)
            | (np.abs(verts[:, 1] - bmin_y) < tol) | (np.abs(verts[:, 1] - bmax_y) < tol)
        ),
    }

    return Mesh2D(
        vertices=verts,
        triangles=triangles,
        boundary_points=bp,
        meta={"type": "structured_rectangle", "nx": nx, "ny": ny},
    )


# ---------------------------------------------------------------------------
# SDF-based mesh (gmsh backend)
# ---------------------------------------------------------------------------

def mesh_sdf_2d(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    bounds_min: Tuple[float, float],
    bounds_max: Tuple[float, float],
    *,
    mesh_size: float = 0.05,
    mesh_size_boundary: Optional[float] = None,
    boundary_n_points: int = 200,
    boundary_tol: float = 0.01,
    algorithm: int = 5,
    optimize_netgen: bool = True,
    boundary_region_samplers: Optional[Dict[str, Callable]] = None,
) -> Optional["Mesh2D"]:
    """Generate a quality 2D mesh inside an SDF domain using gmsh.

    Points on the zero-level-set are extracted and used to define the
    boundary polygon, then gmsh triangulates the interior.

    Parameters
    ----------
    sdf_fn : callable (N,2) -> (N,) signed distance (negative inside)
    bounds_min, bounds_max : bounding box for sampling
    mesh_size : target element size in the interior
    mesh_size_boundary : target element size on boundary (default = mesh_size)
    boundary_n_points : number of boundary points to extract
    boundary_tol : SDF tolerance for boundary extraction
    algorithm : gmsh 2D algorithm (5=Delaunay, 6=Frontal-Delaunay)
    optimize_netgen : run Netgen optimization pass

    Returns Mesh2D or None if gmsh not available.
    """
    try:
        import gmsh
    except ImportError:
        return None

    if mesh_size_boundary is None:
        mesh_size_boundary = mesh_size

    rng = np.random.default_rng(42)
    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)

    # Extract boundary contour via rejection sampling + ordering
    collected = []
    while len(collected) < boundary_n_points * 2:
        cands = bmin + rng.random((boundary_n_points * 50, 2)) * (bmax - bmin)
        d = np.abs(sdf_fn(cands))
        pts = cands[d < boundary_tol]
        if len(pts) > 0:
            collected.extend(pts.tolist())

    if len(collected) < 10:
        return None

    bpts = np.array(collected[:boundary_n_points * 2])
    # Order by angle from centroid
    centroid = bpts.mean(axis=0)
    angles = np.arctan2(bpts[:, 1] - centroid[1], bpts[:, 0] - centroid[0])
    bpts = bpts[np.argsort(angles)]

    # Deduplicate close points
    mask = np.ones(len(bpts), dtype=bool)
    for i in range(1, len(bpts)):
        if np.linalg.norm(bpts[i] - bpts[i - 1]) < mesh_size * 0.3:
            mask[i] = False
    bpts = bpts[mask]

    if len(bpts) < 4:
        return None

    # Use gmsh to mesh
    try:
        gmsh.initialize()
        gmsh.model.add("sdf_mesh")
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)

        # Add boundary points
        point_tags = []
        for pt in bpts:
            tag = gmsh.model.geo.addPoint(float(pt[0]), float(pt[1]), 0.0, meshSize=mesh_size_boundary)
            point_tags.append(tag)

        # Create closed spline / polygon
        line_tags = []
        n_pts = len(point_tags)
        for i in range(n_pts):
            j = (i + 1) % n_pts
            tag = gmsh.model.geo.addLine(point_tags[i], point_tags[j])
            line_tags.append(tag)

        loop = gmsh.model.geo.addCurveLoop(line_tags)
        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        if optimize_netgen:
            gmsh.option.setNumber("Mesh.Optimize", 1)

        gmsh.model.mesh.generate(2)

        # Extract mesh data
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)[:, :2].astype(np.float32)

        elem_types, elem_tags, elem_verts = gmsh.model.mesh.getElements(2, surf)
        triangles = None
        for et, ev in zip(elem_types, elem_verts):
            if et == 2:  # triangle
                triangles = np.array(ev, dtype=np.int32).reshape(-1, 3) - 1

        gmsh.finalize()

        if triangles is None or len(triangles) == 0:
            return None

        return Mesh2D(
            vertices=coords,
            triangles=triangles,
            meta={"type": "sdf_gmsh", "mesh_size": mesh_size, "n_boundary_pts": len(bpts)},
        )
    except Exception:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Polygon triangulation (scipy fallback)
# ---------------------------------------------------------------------------

def mesh_polygon_2d(
    vertices: np.ndarray,
    *,
    mesh_size: Optional[float] = None,
    holes: Optional[List[np.ndarray]] = None,
) -> Optional["Mesh2D"]:
    """Triangulate a 2D polygon using scipy.spatial.Delaunay (basic) or
    the `triangle` library (quality mesh, if available).

    Parameters
    ----------
    vertices : (N, 2) ordered polygon vertices
    mesh_size : target element size (only used with `triangle` library)
    holes : list of (K, 2) polygon arrays for holes

    Returns Mesh2D or None if no triangulation is possible.
    """
    verts = np.asarray(vertices, dtype=np.float64)

    # Try `triangle` library first (quality mesh)
    try:
        import triangle as tr
        seg = np.array([[i, (i + 1) % len(verts)] for i in range(len(verts))])
        data = {"vertices": verts, "segments": seg}
        if holes is not None:
            hole_pts = []
            for h in holes:
                ha = np.asarray(h, dtype=np.float64)
                hole_pts.append(ha.mean(axis=0))
                seg_h = np.array([[len(verts) + len(seg) + i, (len(verts) + len(seg) + i + 1) % len(ha) + len(verts) + len(seg)] for i in range(len(ha))])
            if hole_pts:
                data["holes"] = np.array(hole_pts)

        quality = "pq30" if mesh_size is None else f"pq30a{mesh_size**2 * 0.5:.6f}"
        result = tr.triangulate(data, quality)
        tri_verts = result["vertices"].astype(np.float32)
        tri_tris = result["triangles"].astype(np.int32)
        return Mesh2D(
            vertices=tri_verts,
            triangles=tri_tris,
            meta={"type": "triangle_quality", "quality": quality},
        )
    except ImportError:
        pass

    # Fallback: scipy Delaunay (no quality control)
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(verts)
        # Keep only triangles inside the polygon
        centroids = verts[tri.simplices].mean(axis=1)
        # Check which centroids are inside using winding
        from matplotlib.path import Path as MPath
        path = MPath(verts)
        inside = path.contains_points(centroids)
        tris_filtered = tri.simplices[inside].astype(np.int32)
        return Mesh2D(
            vertices=verts.astype(np.float32),
            triangles=tris_filtered,
            meta={"type": "scipy_delaunay"},
        )
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def mesh_quality_report(mesh: Mesh2D) -> Dict[str, Any]:
    """Generate a comprehensive mesh quality report."""
    min_angs = mesh.min_angles()
    areas = mesh.triangle_areas()
    aspect_ratios = _compute_aspect_ratios(mesh)

    poor_threshold = 20.0  # degrees
    n_poor = int(np.sum(min_angs < poor_threshold))

    return {
        "n_vertices": mesh.n_vertices,
        "n_triangles": mesh.n_triangles,
        "total_area": float(np.sum(areas)),
        "min_angle": {
            "min": float(np.min(min_angs)),
            "max": float(np.max(min_angs)),
            "mean": float(np.mean(min_angs)),
        },
        "aspect_ratio": {
            "min": float(np.min(aspect_ratios)),
            "max": float(np.max(aspect_ratios)),
            "mean": float(np.mean(aspect_ratios)),
        },
        "area": {
            "min": float(np.min(areas)),
            "max": float(np.max(areas)),
            "mean": float(np.mean(areas)),
            "std": float(np.std(areas)),
        },
        "poor_elements": n_poor,
        "poor_fraction": float(n_poor / max(1, len(min_angs))),
    }


def _compute_aspect_ratios(mesh: Mesh2D) -> np.ndarray:
    """Compute aspect ratio = circumradius / (2 * inradius) for each triangle."""
    v = mesh.vertices
    t = mesh.triangles
    a = v[t[:, 1]] - v[t[:, 0]]
    b = v[t[:, 2]] - v[t[:, 1]]
    c = v[t[:, 0]] - v[t[:, 2]]
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    area = 0.5 * np.abs(a[:, 0] * (-c[:, 1]) - a[:, 1] * (-c[:, 0]))
    s = (la + lb + lc) * 0.5
    inradius = area / (s + 1e-12)
    circumradius = (la * lb * lc) / (4.0 * area + 1e-12)
    return circumradius / (2.0 * inradius + 1e-12)
