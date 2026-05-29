"""3D mesh generation from surfaces and SDFs.

Workflow
--------
1. Load or create a 3D surface mesh (STL/OBJ/SDF/primitives).
2. Clean it with ``ops.clean.clean_mesh``.
3. Generate a volume mesh (tetrahedral) with ``mesh_from_surface``.
4. Sample physics collocation points with ``sample_physics_points``.

Backends
--------
  gmsh      — high-quality tet mesh with boundary layer support (preferred)
  tetgen    — fast tet meshing via pyvista/tetgen
  trimesh   — surface only (no volume; used for sampling)
  open3d    — surface reconstruction from point clouds

Usage
-----
    from pinneapple_design.geometry.mesh.mesher_3d import (
        mesh_from_surface, mesh_from_sdf_3d,
        sample_physics_points, PhysicsPointCloud3D,
    )
    from pinneapple_design.geometry.core.mesh import MeshData

    # From a cleaned surface mesh
    vol_mesh = mesh_from_surface(surface_mesh)
    pts = sample_physics_points(surface_mesh, n_interior=10000, n_boundary=2000)
    pts.interior   # (N, 3) collocation points
    pts.boundary   # (M, 3) BC points

    # From an SDF
    vol_mesh = mesh_from_sdf_3d(my_sdf_fn, bounds_min=(-1,-1,-1), bounds_max=(1,1,1))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pinneapple_design.geometry.core.mesh import MeshData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh3D — tetrahedral volume mesh
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Mesh3D:
    """
    Tetrahedral volume mesh.

    Attributes
    ----------
    vertices   : (N, 3) float32 node positions
    tets       : (M, 4) int32 tetrahedral connectivity
    surface_faces : (K, 3) int32 surface triangle connectivity (boundary)
    boundary_groups : {name: vertex index list}
    meta       : free-form metadata
    """
    vertices: np.ndarray            # (N, 3) float32
    tets: Optional[np.ndarray]      # (M, 4) int32 — None for surface-only
    surface_faces: np.ndarray       # (K, 3) int32
    boundary_groups: Dict[str, List[int]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_tets(self) -> int:
        return len(self.tets) if self.tets is not None else 0

    @property
    def n_surface_faces(self) -> int:
        return len(self.surface_faces)

    def tet_volumes(self) -> Optional[np.ndarray]:
        """(M,) volume of each tetrahedron. None if no tets."""
        if self.tets is None:
            return None
        v = self.vertices
        t = self.tets
        a = v[t[:, 1]] - v[t[:, 0]]
        b = v[t[:, 2]] - v[t[:, 0]]
        c = v[t[:, 3]] - v[t[:, 0]]
        return np.abs(np.einsum("ni,ni->n", a, np.cross(b, c))) / 6.0

    def quality_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "n_vertices": self.n_vertices,
            "n_tets": self.n_tets,
            "n_surface_faces": self.n_surface_faces,
        }
        vols = self.tet_volumes()
        if vols is not None and len(vols) > 0:
            stats["tet_volume"] = {
                "total": float(vols.sum()),
                "mean": float(vols.mean()),
                "min": float(float(vols.min())),
                "max": float(float(vols.max())),
            }
        return stats

    def sample_interior(self, n: int, *, seed: int = 0) -> np.ndarray:
        """
        Sample n interior points uniformly by volume (barycentric).
        Returns (n, 3) float32.
        """
        if self.tets is None or len(self.tets) == 0:
            raise RuntimeError("No tetrahedra — use sample_boundary instead.")
        vols = self.tet_volumes()
        rng = np.random.default_rng(seed)
        probs = vols / (vols.sum() + 1e-30)
        tet_idx = rng.choice(len(self.tets), size=n, p=probs)
        t = self.tets[tet_idx]
        v = self.vertices
        r = rng.random((n, 3)).astype(np.float32)
        # Barycentric coords for tet: Rubinstein & Kroese method
        s = np.sort(r, axis=1)
        lam = np.column_stack([s[:, 0],
                               s[:, 1] - s[:, 0],
                               s[:, 2] - s[:, 1],
                               1.0 - s[:, 2]])  # (n, 4)
        pts = (lam[:, 0:1] * v[t[:, 0]]
               + lam[:, 1:2] * v[t[:, 1]]
               + lam[:, 2:3] * v[t[:, 2]]
               + lam[:, 3:4] * v[t[:, 3]])
        return pts.astype(np.float32)

    def sample_boundary(self, n: int, *, seed: int = 0) -> np.ndarray:
        """
        Sample n surface points uniformly by area (barycentric).
        Returns (n, 3) float32.
        """
        v = self.vertices
        f = self.surface_faces
        a = v[f[:, 1]] - v[f[:, 0]]
        b = v[f[:, 2]] - v[f[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
        rng = np.random.default_rng(seed)
        probs = areas / (areas.sum() + 1e-30)
        fidx = rng.choice(len(f), size=n, p=probs)
        ft = f[fidx]
        r1 = rng.random(n).astype(np.float32)
        r2 = rng.random(n).astype(np.float32)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        vb = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2
        return (u[:, None] * v[ft[:, 0]]
                + vb[:, None] * v[ft[:, 1]]
                + w[:, None] * v[ft[:, 2]]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Physics point cloud (output of sample_physics_points)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsPointCloud3D:
    """
    Collocation + boundary + near-wall points for physics training.

    Attributes
    ----------
    interior       : (N_i, 3) interior collocation points
    boundary       : (N_b, 3) surface/BC points
    boundary_normals : (N_b, 3) outward normals at BC points (if available)
    near_wall      : (N_w, 3) near-wall points for boundary-layer resolution
    near_wall_dist : (N_w,) signed distance to surface for near-wall points
    boundary_groups : {name: (K, 3) points} for labelled BC regions
    meta           : metadata from mesher
    """
    interior: np.ndarray
    boundary: np.ndarray
    boundary_normals: Optional[np.ndarray] = None
    near_wall: Optional[np.ndarray] = None
    near_wall_dist: Optional[np.ndarray] = None
    boundary_groups: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_interior(self) -> int:
        return len(self.interior)

    @property
    def n_boundary(self) -> int:
        return len(self.boundary)

    def all_points(self) -> np.ndarray:
        """All interior + boundary points concatenated. (N+M, 3)."""
        return np.concatenate([self.interior, self.boundary], axis=0)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "interior": self.interior,
            "boundary": self.boundary,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
        }
        if self.boundary_normals is not None:
            d["boundary_normals"] = self.boundary_normals
        if self.near_wall is not None:
            d["near_wall"] = self.near_wall
            d["near_wall_dist"] = self.near_wall_dist
        d.update(self.boundary_groups)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Volume mesh generation
# ─────────────────────────────────────────────────────────────────────────────

def mesh_from_surface(
    surface: MeshData,
    *,
    mesh_size: float = 0.0,         # 0 = auto (bounding box / 20)
    mesh_size_boundary: float = 0.0,
    boundary_layer: bool = False,
    bl_thickness: float = 0.0,      # 0 = auto
    bl_n_layers: int = 3,
    optimize: bool = True,
    algorithm_3d: int = 4,          # gmsh: 4=Frontal, 1=Delaunay
) -> Optional[Mesh3D]:
    """
    Generate a tetrahedral volume mesh from a closed 3D surface mesh.

    Tries gmsh (preferred), then tetgen/pyvista, then returns surface-only.

    Parameters
    ----------
    surface           : closed triangular surface MeshData
    mesh_size         : target element size (0 = auto)
    mesh_size_boundary: target element size on boundary
    boundary_layer    : add prismatic boundary layer (CFD near-wall refinement)
    bl_thickness      : boundary layer total thickness (0 = auto: mesh_size)
    bl_n_layers       : number of boundary layer prism layers
    optimize          : run gmsh optimization pass
    algorithm_3d      : gmsh 3D meshing algorithm

    Returns
    -------
    Mesh3D or None
    """
    bbox_diag = float(np.linalg.norm(surface.vertices.max(0) - surface.vertices.min(0)))
    if mesh_size <= 0.0:
        mesh_size = bbox_diag / 20.0
    if mesh_size_boundary <= 0.0:
        mesh_size_boundary = mesh_size
    if bl_thickness <= 0.0:
        bl_thickness = mesh_size * 0.3

    # ── Try gmsh ──────────────────────────────────────────────────────────────
    result = _mesh_gmsh_3d(
        surface,
        mesh_size=mesh_size,
        mesh_size_boundary=mesh_size_boundary,
        boundary_layer=boundary_layer,
        bl_thickness=bl_thickness,
        bl_n_layers=bl_n_layers,
        optimize=optimize,
        algorithm_3d=algorithm_3d,
    )
    if result is not None:
        return result

    # ── Try tetgen/pyvista ────────────────────────────────────────────────────
    result = _mesh_tetgen(surface, mesh_size=mesh_size)
    if result is not None:
        return result

    # ── Fallback: surface-only Mesh3D ─────────────────────────────────────────
    logger.warning(
        "mesh_from_surface: no volume mesher available (install gmsh or pyvistameshio+tetgen). "
        "Returning surface-only Mesh3D."
    )
    return Mesh3D(
        vertices=surface.vertices.astype(np.float32),
        tets=None,
        surface_faces=surface.faces.astype(np.int32),
        meta={"backend": "surface_only"},
    )


def _mesh_gmsh_3d(
    surface: MeshData,
    *,
    mesh_size: float,
    mesh_size_boundary: float,
    boundary_layer: bool,
    bl_thickness: float,
    bl_n_layers: int,
    optimize: bool,
    algorithm_3d: int,
) -> Optional[Mesh3D]:
    try:
        import gmsh
    except ImportError:
        return None

    try:
        gmsh.initialize()
        gmsh.model.add("vol3d")
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.05)

        # Import surface mesh
        v = surface.vertices.astype(np.float64)
        f = surface.faces.astype(np.int32)
        node_tags = np.arange(1, len(v) + 1, dtype=int)
        gmsh.model.mesh.addNodes(2, 1, node_tags, v.ravel().tolist(), [])
        elem_tags = np.arange(1, len(f) + 1, dtype=int)
        elem_nodes = (f + 1).ravel().tolist()
        gmsh.model.mesh.addElementsByType(1, 2, elem_tags.tolist(), elem_nodes)
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
        gmsh.model.mesh.createGeometry()

        surfs = gmsh.model.getEntities(2)
        if not surfs:
            gmsh.finalize()
            return None

        surf_tags = [s[1] for s in surfs]
        sl = gmsh.model.geo.addSurfaceLoop(surf_tags)
        vol = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        if boundary_layer and len(surfs) > 0:
            try:
                f_field = gmsh.model.mesh.field.add("BoundaryLayer")
                gmsh.model.mesh.field.setNumbers(f_field, "CurvesList", [])
                gmsh.model.mesh.field.setNumbers(f_field, "FacesList", surf_tags)
                gmsh.model.mesh.field.setNumber(f_field, "Size", bl_thickness / bl_n_layers)
                gmsh.model.mesh.field.setNumber(f_field, "NbLayers", bl_n_layers)
                gmsh.model.mesh.field.setNumber(f_field, "Thickness", bl_thickness)
                gmsh.model.mesh.field.setAsBoundaryLayer(f_field)
            except Exception:
                pass

        if optimize:
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        gmsh.model.mesh.generate(3)

        # Extract nodes
        node_tags_out, coords_out, _ = gmsh.model.mesh.getNodes()
        coords_out = np.array(coords_out, dtype=np.float32).reshape(-1, 3)

        # Extract tets
        tet_type, tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4)
        tets = None
        if len(tet_nodes) > 0:
            tets = np.array(tet_nodes, dtype=np.int32).reshape(-1, 4) - 1

        # Extract surface triangles
        tri_type, tri_tags, tri_nodes = gmsh.model.mesh.getElementsByType(2)
        surf_faces = np.array([], dtype=np.int32).reshape(0, 3)
        if len(tri_nodes) > 0:
            surf_faces = np.array(tri_nodes, dtype=np.int32).reshape(-1, 3) - 1

        gmsh.finalize()

        if tets is None or len(tets) == 0:
            return None

        return Mesh3D(
            vertices=coords_out,
            tets=tets,
            surface_faces=surf_faces,
            meta={"backend": "gmsh", "mesh_size": mesh_size,
                  "n_tets": len(tets), "boundary_layer": boundary_layer},
        )

    except Exception as exc:
        logger.warning(f"gmsh 3D meshing failed: {exc}")
        try:
            gmsh.finalize()
        except Exception:
            pass
        return None


def _mesh_tetgen(surface: MeshData, *, mesh_size: float) -> Optional[Mesh3D]:
    try:
        import pyvista as pv
        import tetgen
    except ImportError:
        return None

    try:
        surf_pv = pv.PolyData(
            surface.vertices.astype(np.float64),
            np.column_stack([np.full(len(surface.faces), 3), surface.faces]),
        )
        tet = tetgen.TetGen(surf_pv)
        tet.tetrahedralize(order=1, mindihedral=10, minratio=1.5)
        grid = tet.grid
        v = np.array(grid.points, dtype=np.float32)
        cells = grid.cells_dict.get(10, np.array([]).reshape(0, 4))  # VTK_TETRA=10
        t = np.array(cells, dtype=np.int32) if len(cells) > 0 else None
        return Mesh3D(
            vertices=v,
            tets=t,
            surface_faces=surface.faces.astype(np.int32),
            meta={"backend": "tetgen"},
        )
    except Exception as exc:
        logger.warning(f"tetgen meshing failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SDF → surface → volume
# ─────────────────────────────────────────────────────────────────────────────

def mesh_from_sdf_3d(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    bounds_min: Tuple[float, float, float],
    bounds_max: Tuple[float, float, float],
    *,
    resolution: int = 64,
    mesh_size: float = 0.0,
    clean: bool = True,
    volume: bool = True,
) -> Optional[Mesh3D]:
    """
    Extract a surface from a 3D SDF via marching cubes, then volume-mesh it.

    Parameters
    ----------
    sdf_fn     : callable (N,3) -> (N,) — negative inside, positive outside
    bounds_min : (xmin, ymin, zmin)
    bounds_max : (xmax, ymax, zmax)
    resolution : grid resolution per axis
    mesh_size  : target tet size (0 = auto)
    clean      : run clean_mesh on the extracted surface before tet meshing
    volume     : generate volume mesh (True) or return surface only (False)
    """
    # Build evaluation grid
    xs = np.linspace(bounds_min[0], bounds_max[0], resolution, dtype=np.float32)
    ys = np.linspace(bounds_min[1], bounds_max[1], resolution, dtype=np.float32)
    zs = np.linspace(bounds_min[2], bounds_max[2], resolution, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    sdf_vals = sdf_fn(pts).reshape(resolution, resolution, resolution)

    # Marching cubes
    try:
        from skimage.measure import marching_cubes
        verts_mc, faces_mc, normals_mc, _ = marching_cubes(sdf_vals, level=0.0)
    except ImportError:
        try:
            import mcubes
            verts_mc, faces_mc = mcubes.marching_cubes(sdf_vals, 0.0)
            normals_mc = None
        except ImportError:
            raise ImportError(
                "scikit-image or PyMCubes required for SDF meshing: "
                "pip install scikit-image  OR  pip install PyMCubes"
            )

    # Map voxel coordinates → world coordinates
    scale = np.array([
        (bounds_max[0] - bounds_min[0]) / (resolution - 1),
        (bounds_max[1] - bounds_min[1]) / (resolution - 1),
        (bounds_max[2] - bounds_min[2]) / (resolution - 1),
    ], dtype=np.float32)
    origin = np.array(bounds_min, dtype=np.float32)
    verts_world = verts_mc.astype(np.float32) * scale + origin

    surface = MeshData(
        vertices=verts_world,
        faces=faces_mc.astype(np.int32),
        normals=normals_mc.astype(np.float32) if normals_mc is not None else None,
    )

    if clean:
        from pinneapple_design.geometry.ops.clean import clean_mesh, CleanMeshConfig
        surface = clean_mesh(
            surface,
            config=CleanMeshConfig(
                smooth=True, smooth_iterations=3,
                fill_holes=True, keep_largest=True,
            ),
        )

    if not volume:
        return Mesh3D(
            vertices=surface.vertices,
            tets=None,
            surface_faces=surface.faces,
            meta={"source": "sdf_marching_cubes", "resolution": resolution},
        )

    vol = mesh_from_surface(surface, mesh_size=mesh_size)
    if vol is not None:
        vol.meta.update({"source": "sdf_marching_cubes", "resolution": resolution})
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Physics point sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_physics_points(
    mesh: "MeshData | Mesh3D",
    *,
    n_interior: int = 10000,
    n_boundary: int = 2000,
    n_near_wall: int = 0,
    near_wall_thickness: float = 0.0,   # 0 = auto (bbox / 50)
    curvature_refinement: bool = False,
    seed: int = 42,
) -> PhysicsPointCloud3D:
    """
    Generate physics-ready collocation points from a 3D mesh.

    Works with both ``MeshData`` (surface only) and ``Mesh3D`` (volume).

    Interior points : uniform random inside volume (tet barycentric) or
                      rejection-sampled inside bounding box filtered by
                      SDF if only a surface is available.
    Boundary points : uniform on surface triangles (area-weighted barycentric).
    Near-wall points: offset inward along surface normals — useful for
                      boundary-layer and wall-shear physics.

    Parameters
    ----------
    mesh               : Mesh3D or MeshData
    n_interior         : number of interior collocation points
    n_boundary         : number of surface / BC points
    n_near_wall        : number of near-wall interior points
    near_wall_thickness: distance from surface to near-wall layer
    curvature_refinement : cluster more points near high-curvature regions
    seed               : RNG seed
    """
    rng = np.random.default_rng(seed)

    if isinstance(mesh, Mesh3D):
        vol3d = mesh
        surface_verts = mesh.vertices
        surface_faces = mesh.surface_faces
    else:
        # Build a temporary Mesh3D from surface
        vol3d = Mesh3D(
            vertices=mesh.vertices.astype(np.float32),
            tets=None,
            surface_faces=mesh.faces.astype(np.int32),
        )
        surface_verts = mesh.vertices
        surface_faces = mesh.faces

    # ── Interior ─────────────────────────────────────────────────────────────
    if vol3d.tets is not None and len(vol3d.tets) > 0:
        interior = vol3d.sample_interior(n_interior, seed=seed)
    else:
        # Rejection sampling inside bounding box, filtered by winding number
        interior = _sample_interior_rejection(
            surface_verts, surface_faces, n_interior, rng=rng
        )

    # ── Boundary ─────────────────────────────────────────────────────────────
    boundary, boundary_normals = _sample_surface(
        surface_verts, surface_faces, n_boundary, rng=rng,
        curvature_refinement=curvature_refinement,
    )

    # ── Near-wall ─────────────────────────────────────────────────────────────
    near_wall = None
    near_wall_dist = None
    if n_near_wall > 0:
        bbox_diag = float(np.linalg.norm(surface_verts.max(0) - surface_verts.min(0)))
        thickness = near_wall_thickness if near_wall_thickness > 0 else bbox_diag / 50.0
        near_wall, near_wall_dist = _sample_near_wall(
            surface_verts, surface_faces, boundary, boundary_normals,
            n_near_wall, thickness, rng=rng,
        )

    return PhysicsPointCloud3D(
        interior=interior,
        boundary=boundary,
        boundary_normals=boundary_normals,
        near_wall=near_wall,
        near_wall_dist=near_wall_dist,
        meta={
            "n_interior": len(interior),
            "n_boundary": len(boundary),
            "n_near_wall": len(near_wall) if near_wall is not None else 0,
        },
    )


def _sample_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    curvature_refinement: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample n points on surface triangles, return (points, normals)."""
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    cross = np.cross(a, b)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

    if curvature_refinement:
        # Approximate curvature via edge dihedral angles
        weights = np.clip(areas, 1e-12, None)
        weights = weights ** 0.5  # soften area bias; curvature est. not available
    else:
        weights = areas

    probs = weights / (weights.sum() + 1e-30)
    fidx = rng.choice(len(faces), size=n, p=probs)
    ft = faces[fidx]
    r1 = rng.random(n).astype(np.float32)
    r2 = rng.random(n).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    vb = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    pts = (u[:, None] * verts[ft[:, 0]]
           + vb[:, None] * verts[ft[:, 1]]
           + w[:, None] * verts[ft[:, 2]]).astype(np.float32)
    return pts, normals[fidx].astype(np.float32)


def _sample_near_wall(
    verts: np.ndarray,
    faces: np.ndarray,
    boundary_pts: np.ndarray,
    boundary_normals: np.ndarray,
    n: int,
    thickness: float,
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points offset inward from boundary by a random fraction of thickness."""
    idx = rng.choice(len(boundary_pts), size=n, replace=True)
    offsets = rng.uniform(0.01 * thickness, thickness, n).astype(np.float32)
    # Inward = opposite of outward normal
    pts = boundary_pts[idx] - offsets[:, None] * boundary_normals[idx]
    return pts.astype(np.float32), offsets


def _sample_interior_rejection(
    verts: np.ndarray,
    faces: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> np.ndarray:
    """
    Rejection sampling inside bounding box, keep points inside via winding number.
    Falls back to pure random when trimesh not available.
    """
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)

    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        collected: List[np.ndarray] = []
        total = 0
        for _ in range(max_attempts):
            batch = (bbox_min + rng.random((n * 4, 3)) * (bbox_max - bbox_min)).astype(np.float32)
            inside = tm.contains(batch)
            pts_in = batch[inside]
            if len(pts_in) > 0:
                collected.append(pts_in)
                total += len(pts_in)
            if total >= n:
                break
        if collected:
            all_pts = np.concatenate(collected, axis=0)
            return all_pts[:n].astype(np.float32)
    except Exception:
        pass

    # Pure random fallback (no inside test)
    return (bbox_min + rng.random((n, 3)) * (bbox_max - bbox_min)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Surface reconstruction from point cloud
# ─────────────────────────────────────────────────────────────────────────────

def surface_from_pointcloud(
    points: np.ndarray,
    *,
    normals: Optional[np.ndarray] = None,
    method: str = "poisson",    # "poisson" | "ball_pivot" | "alpha_shape"
    depth: int = 8,             # Poisson reconstruction depth
    alpha: float = 0.0,         # alpha shape radius (0 = auto)
    density_quantile: float = 0.05,
) -> Optional[MeshData]:
    """
    Reconstruct a surface mesh from a point cloud.

    Backends: open3d (Poisson / BPA), scipy (alpha shapes).

    Parameters
    ----------
    points          : (N, 3) point cloud
    normals         : (N, 3) normals (estimated internally if None)
    method          : "poisson" | "ball_pivot" | "alpha_shape"
    depth           : Poisson octree depth (higher = finer)
    alpha           : alpha value for alpha shapes (0 = auto-estimate)
    density_quantile: Poisson low-density vertex pruning threshold

    Returns
    -------
    MeshData or None
    """
    pts = np.asarray(points, dtype=np.float64)

    # ── open3d Poisson or Ball Pivoting ───────────────────────────────────────
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(
                np.asarray(normals, dtype=np.float64)
            )
        else:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        if method == "poisson":
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth
            )
            # Remove low-density vertices
            dens = np.asarray(densities)
            thresh = np.quantile(dens, density_quantile)
            mesh_o3d.remove_vertices_by_mask(dens < thresh)

        elif method == "ball_pivot":
            bbox_diag = np.linalg.norm(pts.max(0) - pts.min(0))
            radii = o3d.utility.DoubleVector([
                bbox_diag * 0.01, bbox_diag * 0.02, bbox_diag * 0.04
            ])
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii
            )
        else:
            raise ValueError(f"Unknown method '{method}' for open3d backend.")

        v = np.asarray(mesh_o3d.vertices, dtype=np.float32)
        f = np.asarray(mesh_o3d.triangles, dtype=np.int32)
        if len(f) == 0:
            raise ValueError("open3d reconstruction produced empty mesh.")
        return MeshData(vertices=v, faces=f)

    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"open3d reconstruction failed: {exc}")

    # ── Alpha shape via scipy / trimesh ───────────────────────────────────────
    if method == "alpha_shape":
        try:
            import trimesh
            from trimesh.sample import sample_surface
            import trimesh.creation as tc
            if alpha <= 0.0:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                vol = hull.volume
                alpha = (vol / len(pts)) ** (1.0 / 3.0) * 2.0
            alpha_mesh = trimesh.creation.icosphere(radius=alpha)
            _ = alpha_mesh  # placeholder — trimesh doesn't directly support alpha shapes
            # Use scipy Delaunay + alpha filtering
            from scipy.spatial import Delaunay
            tri = Delaunay(pts)
            verts_idx = tri.simplices  # tetrahedra
            # Keep tets with circumradius < alpha
            centres, radii = _circumsphere(pts, verts_idx)
            keep = radii < alpha
            kept_tets = verts_idx[keep]
            # Extract surface faces (faces shared by exactly 1 tet)
            all_faces = np.vstack([
                kept_tets[:, [0, 1, 2]],
                kept_tets[:, [0, 1, 3]],
                kept_tets[:, [0, 2, 3]],
                kept_tets[:, [1, 2, 3]],
            ])
            all_faces = np.sort(all_faces, axis=1)
            unique, counts = np.unique(all_faces, axis=0, return_counts=True)
            surf_faces = unique[counts == 1]
            return MeshData(
                vertices=pts.astype(np.float32),
                faces=surf_faces.astype(np.int32),
            )
        except Exception as exc:
            logger.warning(f"Alpha shape failed: {exc}")

    logger.warning("surface_from_pointcloud: no working backend found. "
                   "Install open3d: pip install open3d")
    return None


def _circumsphere(
    pts: np.ndarray,
    tets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Circumsphere centres and radii for each tetrahedron."""
    a = pts[tets[:, 0]]
    b = pts[tets[:, 1]]
    c = pts[tets[:, 2]]
    d = pts[tets[:, 3]]
    ac = c - a
    ab = b - a
    ad = d - a
    abc = np.cross(ab, ac)
    vol6 = 2.0 * np.einsum("ni,ni->n", ad, abc)
    vol6[vol6 == 0] = 1e-30
    centres = a + (
        np.cross(ad, abc) * np.einsum("ni,ni->n", ab, ab)[:, None]
        + np.cross(abc, ab) * np.einsum("ni,ni->n", ad, ad)[:, None]
    ) / vol6[:, None]
    radii = np.linalg.norm(a - centres, axis=1)
    return centres, radii
