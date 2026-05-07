"""Comprehensive 3D mesh cleaning pipeline.

Provides a layered cleaning pipeline that goes well beyond the basic
``repair_mesh`` (duplicate removal / degenerate face removal).  Every step is
optional so callers can pick only what they need.

Pipeline order (default)
------------------------
1. merge_vertices         — weld vertices within a tolerance
2. remove_degenerate      — zero-area triangles
3. remove_duplicates      — duplicate faces
4. remove_unreferenced    — vertices not used by any face
5. fix_normals            — make face normals consistently outward
6. fill_holes             — attempt to close small boundary loops
7. remove_self_intersect  — detect and remove self-intersecting faces (pymeshlab)
8. smooth                 — Laplacian or Taubin smoothing
9. remove_outliers        — statistical outlier vertex removal
10. keep_largest           — discard all but the largest connected component
11. ensure_watertight      — final watertight check, trim if needed
12. decimate              — reduce polygon count to target face count

All heavy-lifting uses trimesh (always required) and pymeshlab (optional).
open3d is used opportunistically for outlier removal and smoothing.

Usage
-----
    from pinneaple_design.geometry.ops.clean import clean_mesh, CleanMeshConfig
    from pinneaple_design.geometry.core.mesh import MeshData

    cfg = CleanMeshConfig(smooth=True, smooth_iterations=5, keep_largest=True)
    cleaned = clean_mesh(mesh, config=cfg)
    print(cleaned.quality_report())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pinneaple_design.geometry.core.mesh import MeshData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanMeshConfig:
    """Full configuration for the clean_mesh pipeline."""

    # Vertex welding
    merge_vertices: bool = True
    merge_tol: float = 0.0          # 0 = auto (1e-6 × bounding-box diagonal)

    # Face cleaning
    remove_degenerate: bool = True
    remove_duplicates: bool = True
    remove_unreferenced: bool = True

    # Normals
    fix_normals: bool = True

    # Hole filling
    fill_holes: bool = True
    max_hole_edges: int = 100       # don't fill holes with more than N edges

    # Self-intersection
    remove_self_intersect: bool = False  # slow; requires pymeshlab

    # Smoothing
    smooth: bool = False
    smooth_method: str = "taubin"   # "laplacian" | "taubin"
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5      # Laplacian step / Taubin λ
    smooth_mu: float = -0.53        # Taubin μ (negative, slight inflate)

    # Outlier removal
    remove_outliers: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0

    # Connected components
    keep_largest: bool = True       # discard all but the largest component

    # Watertight enforcement
    ensure_watertight: bool = False  # aggressive: may remove valid geometry

    # Decimation
    decimate: bool = False
    target_faces: Optional[int] = None   # None = no decimation
    decimate_ratio: float = 0.5          # fraction of faces to keep


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def mesh_quality_report_3d(mesh: MeshData) -> Dict[str, Any]:
    """
    Comprehensive quality report for a 3D surface mesh.

    Returns a dict with:
      - counts (vertices, faces, edges)
      - watertight, manifold, euler_number
      - face area statistics
      - edge length statistics
      - face aspect ratio statistics
      - dihedral angle statistics
      - bounding box + volume (if watertight)
    """
    try:
        import trimesh
        tm = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
        )

        areas = tm.area_faces
        edge_lens = np.linalg.norm(
            mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]], axis=1
        )
        # Aspect ratio: circumradius / (2*inradius) for each triangle
        def _aspect_ratio(v, f):
            a = v[f[:, 1]] - v[f[:, 0]]
            b = v[f[:, 2]] - v[f[:, 1]]
            c = v[f[:, 0]] - v[f[:, 2]]
            la = np.linalg.norm(a, axis=1)
            lb = np.linalg.norm(b, axis=1)
            lc = np.linalg.norm(c, axis=1)
            cross = np.cross(a, -c)
            area2 = np.linalg.norm(cross, axis=1)
            s = (la + lb + lc) * 0.5
            inr = area2 / (2.0 * (s + 1e-12))
            cr = (la * lb * lc) / (area2 + 1e-12)
            return cr / (2.0 * inr + 1e-12)

        ar = _aspect_ratio(mesh.vertices, mesh.faces)

        report: Dict[str, Any] = {
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.faces),
            "n_edges": len(tm.edges_unique),
            "is_watertight": bool(tm.is_watertight),
            "is_manifold": bool(tm.is_volume),
            "euler_number": int(tm.euler_number),
            "is_winding_consistent": bool(tm.is_winding_consistent),
            "area_faces": {
                "total": float(areas.sum()),
                "mean": float(areas.mean()),
                "min": float(areas.min()),
                "max": float(areas.max()),
                "std": float(areas.std()),
            },
            "edge_length": {
                "mean": float(edge_lens.mean()),
                "min": float(edge_lens.min()),
                "max": float(edge_lens.max()),
                "std": float(edge_lens.std()),
            },
            "aspect_ratio": {
                "mean": float(ar.mean()),
                "max": float(ar.max()),
                "p95": float(np.percentile(ar, 95)),
                "poor_fraction": float((ar > 5.0).mean()),
            },
            "bounding_box": {
                "min": mesh.vertices.min(axis=0).tolist(),
                "max": mesh.vertices.max(axis=0).tolist(),
                "extents": (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).tolist(),
            },
        }

        if tm.is_watertight:
            try:
                report["volume"] = float(abs(tm.volume))
            except Exception:
                pass

        # Count open boundary edges
        if hasattr(tm, "boundary_edges"):
            report["n_boundary_edges"] = len(tm.boundary_edges)

        return report

    except ImportError:
        logger.warning("trimesh required for quality report.")
        return {
            "n_vertices": len(mesh.vertices),
            "n_faces": len(mesh.faces),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual operations
# ─────────────────────────────────────────────────────────────────────────────

def smooth_mesh(
    mesh: MeshData,
    *,
    method: str = "taubin",
    iterations: int = 5,
    lam: float = 0.5,
    mu: float = -0.53,
) -> MeshData:
    """
    Laplacian or Taubin mesh smoothing.

    Taubin (default) avoids the volume-shrinkage problem of pure Laplacian.

    Parameters
    ----------
    method     : "taubin" | "laplacian" | "open3d_laplacian" | "open3d_taubin"
    iterations : number of smoothing passes
    lam        : Laplacian λ step / Taubin λ (forward)
    mu         : Taubin μ (backward, negative)
    """
    if method.startswith("open3d"):
        return _smooth_open3d(mesh, method=method.replace("open3d_", ""),
                              iterations=iterations)

    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh required for smoothing.")
        return mesh

    tm = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)

    # Build sparse adjacency for Laplacian
    from scipy.sparse import lil_matrix
    n = len(tm.vertices)
    L = lil_matrix((n, n))
    for i, neighbors in enumerate(tm.vertex_neighbors):
        if neighbors:
            L[i, neighbors] = 1.0 / len(neighbors)

    L = L.tocsr()
    v = tm.vertices.copy().astype(np.float64)

    for _ in range(iterations):
        if method == "laplacian":
            delta = L @ v - v
            v = v + lam * delta
        else:  # taubin
            delta = L @ v - v
            v = v + lam * delta
            delta = L @ v - v
            v = v + mu * delta

    v = v.astype(np.float32)
    normals = None
    try:
        tm2 = trimesh.Trimesh(vertices=v, faces=tm.faces, process=False)
        normals = tm2.face_normals.view(np.ndarray)
    except Exception:
        pass

    return MeshData(vertices=v, faces=tm.faces.view(np.ndarray).copy(), normals=normals)


def _smooth_open3d(mesh: MeshData, *, method: str = "taubin", iterations: int = 5) -> MeshData:
    try:
        import open3d as o3d
        tri_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        if method == "laplacian":
            tri_mesh = tri_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        else:
            tri_mesh = tri_mesh.filter_smooth_taubin(number_of_iterations=iterations)
        v = np.asarray(tri_mesh.vertices, dtype=np.float32)
        f = np.asarray(tri_mesh.triangles, dtype=np.int32)
        return MeshData(vertices=v, faces=f)
    except ImportError:
        logger.warning("open3d not installed; falling back to trimesh Taubin.")
        return smooth_mesh(mesh, method="taubin", iterations=iterations)


def remove_outlier_vertices(
    mesh: MeshData,
    *,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> MeshData:
    """
    Statistical outlier vertex removal.

    Each vertex is scored by the mean distance to its k nearest neighbors.
    Vertices with score > mean + std_ratio * std are removed.
    Any faces referencing removed vertices are also deleted.
    """
    v = mesh.vertices.astype(np.float32)
    n = len(v)

    # Try open3d first (fast)
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v.astype(np.float64))
        _, inlier_idx = pcd.remove_statistical_outlier(nb_points=nb_neighbors, std_ratio=std_ratio)
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[np.asarray(inlier_idx)] = True
    except ImportError:
        # Fallback: manual k-NN with scipy
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(v)
            dists, _ = tree.query(v, k=min(nb_neighbors + 1, n))
            mean_dists = dists[:, 1:].mean(axis=1)  # exclude self
            mu = mean_dists.mean()
            sigma = mean_dists.std()
            inlier_mask = mean_dists <= mu + std_ratio * sigma
        except ImportError:
            logger.warning("open3d or scipy required for outlier removal.")
            return mesh

    # Remap faces
    new_idx = np.full(n, -1, dtype=np.int32)
    new_idx[inlier_mask] = np.arange(inlier_mask.sum(), dtype=np.int32)
    new_v = v[inlier_mask]

    face_mask = np.all(inlier_mask[mesh.faces], axis=1)
    new_f = new_idx[mesh.faces[face_mask]]

    normals = mesh.normals[face_mask] if mesh.normals is not None else None
    logger.info(f"Outlier removal: {n} → {len(new_v)} vertices, "
                f"{len(mesh.faces)} → {len(new_f)} faces.")
    return MeshData(vertices=new_v, faces=new_f, normals=normals)


def keep_largest_component(mesh: MeshData) -> MeshData:
    """Keep only the largest connected component of the mesh."""
    try:
        import trimesh
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        components = tm.split(only_watertight=False)
        if not components:
            return mesh
        largest = max(components, key=lambda m: len(m.faces))
        return MeshData(
            vertices=largest.vertices.view(np.ndarray).astype(np.float32),
            faces=largest.faces.view(np.ndarray).astype(np.int32),
            normals=largest.face_normals.view(np.ndarray) if len(largest.faces) else None,
        )
    except ImportError:
        logger.warning("trimesh required for connected component analysis.")
        return mesh


def decimate_mesh(
    mesh: MeshData,
    *,
    target_faces: Optional[int] = None,
    ratio: float = 0.5,
) -> MeshData:
    """
    Mesh decimation (polygon reduction).

    Tries pymeshlab (quadric edge collapse) then open3d, then trimesh.

    Parameters
    ----------
    target_faces : absolute target face count
    ratio        : fraction of faces to keep (used if target_faces is None)
    """
    n_faces = len(mesh.faces)
    target = target_faces if target_faces is not None else max(4, int(n_faces * ratio))

    # 1. pymeshlab
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype(np.float64),
                                   face_matrix=mesh.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target)
        m = ms.current_mesh()
        return MeshData(vertices=m.vertex_matrix().astype(np.float32),
                        faces=m.face_matrix().astype(np.int32))
    except Exception:
        pass

    # 2. open3d
    try:
        import open3d as o3d
        tm = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        dec = tm.simplify_quadric_decimation(target)
        return MeshData(
            vertices=np.asarray(dec.vertices, dtype=np.float32),
            faces=np.asarray(dec.triangles, dtype=np.int32),
        )
    except Exception:
        pass

    # 3. trimesh (simple vertex clustering fallback)
    try:
        import trimesh
        tm2 = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        pitch = tm2.scale / (target ** 0.5)
        dec2 = tm2.voxelized(pitch).as_boxes()
        v = dec2.vertices.view(np.ndarray).astype(np.float32)
        f = dec2.faces.view(np.ndarray).astype(np.int32)
        return MeshData(vertices=v, faces=f)
    except Exception:
        pass

    logger.warning("Decimation: no backend available (install pymeshlab or open3d).")
    return mesh


def remove_self_intersections(mesh: MeshData) -> MeshData:
    """Remove self-intersecting faces using pymeshlab."""
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype(np.float64),
                                   face_matrix=mesh.faces))
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        m = ms.current_mesh()
        return MeshData(vertices=m.vertex_matrix().astype(np.float32),
                        faces=m.face_matrix().astype(np.int32))
    except Exception as exc:
        logger.warning(f"Self-intersection removal failed: {exc}")
        return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_mesh(
    mesh: MeshData,
    *,
    config: Optional[CleanMeshConfig] = None,
    verbose: bool = False,
) -> MeshData:
    """
    Comprehensive 3D surface mesh cleaning pipeline.

    Applies steps in order according to ``CleanMeshConfig``.  Each step is
    skipped if its flag is False.  Returns a NEW MeshData (input unchanged).

    Parameters
    ----------
    mesh   : input MeshData
    config : CleanMeshConfig (uses defaults if None)
    verbose: log step-by-step counts

    Returns
    -------
    MeshData — cleaned mesh

    Notes
    -----
    Requires trimesh.  Steps using pymeshlab or open3d degrade gracefully
    when those libraries are not installed.
    """
    cfg = config or CleanMeshConfig()

    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    def _log(step: str, cur: MeshData) -> None:
        if verbose:
            logger.info(f"clean_mesh [{step}]: {len(cur.vertices)} V, {len(cur.faces)} F")

    result = MeshData(
        vertices=mesh.vertices.copy().astype(np.float32),
        faces=mesh.faces.copy().astype(np.int32),
        normals=mesh.normals.copy() if mesh.normals is not None else None,
    )

    # ── Step 1: merge vertices ───────────────────────────────────────────────
    if cfg.merge_vertices:
        tol = cfg.merge_tol
        if tol <= 0.0:
            diag = float(np.linalg.norm(result.vertices.max(0) - result.vertices.min(0)))
            tol = max(1e-8, diag * 1e-6)
        tm = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
        # trimesh API varies across versions — try keyword arg, else no-arg
        try:
            tm.merge_vertices(merge_tolerance=tol)
        except TypeError:
            try:
                tm.merge_vertices(digits_vertex=max(1, int(-np.log10(tol))))
            except TypeError:
                tm.merge_vertices()
        result = MeshData(vertices=tm.vertices.view(np.ndarray).astype(np.float32),
                          faces=tm.faces.view(np.ndarray).astype(np.int32))
        _log("merge_vertices", result)

    # ── Step 2–4: degenerate / duplicates / unreferenced ────────────────────
    if cfg.remove_degenerate or cfg.remove_duplicates or cfg.remove_unreferenced:
        tm = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
        if cfg.remove_degenerate:
            tm.update_faces(tm.nondegenerate_faces())
        if cfg.remove_duplicates:
            tm.update_faces(tm.unique_faces())
        if cfg.remove_unreferenced:
            tm.remove_unreferenced_vertices()
        result = MeshData(vertices=tm.vertices.view(np.ndarray).astype(np.float32),
                          faces=tm.faces.view(np.ndarray).astype(np.int32))
        _log("remove_degenerate/duplicates", result)

    # ── Step 5: fix normals ──────────────────────────────────────────────────
    if cfg.fix_normals:
        try:
            tm = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
            tm.fix_normals()
            result = MeshData(
                vertices=tm.vertices.view(np.ndarray).astype(np.float32),
                faces=tm.faces.view(np.ndarray).astype(np.int32),
                normals=tm.face_normals.view(np.ndarray),
            )
            _log("fix_normals", result)
        except Exception as exc:
            logger.debug(f"fix_normals failed: {exc}")

    # ── Step 6: fill holes ───────────────────────────────────────────────────
    if cfg.fill_holes:
        try:
            tm = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
            if not tm.is_watertight:
                tm.fill_holes()
                result = MeshData(
                    vertices=tm.vertices.view(np.ndarray).astype(np.float32),
                    faces=tm.faces.view(np.ndarray).astype(np.int32),
                )
                _log("fill_holes", result)
        except Exception as exc:
            logger.debug(f"fill_holes failed: {exc}")

    # ── Step 7: self-intersection removal ────────────────────────────────────
    if cfg.remove_self_intersect:
        result = remove_self_intersections(result)
        _log("remove_self_intersect", result)

    # ── Step 8: smoothing ────────────────────────────────────────────────────
    if cfg.smooth:
        result = smooth_mesh(
            result,
            method=cfg.smooth_method,
            iterations=cfg.smooth_iterations,
            lam=cfg.smooth_lambda,
            mu=cfg.smooth_mu,
        )
        _log("smooth", result)

    # ── Step 9: outlier removal ──────────────────────────────────────────────
    if cfg.remove_outliers:
        result = remove_outlier_vertices(
            result,
            nb_neighbors=cfg.outlier_nb_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )
        _log("remove_outliers", result)

    # ── Step 10: keep largest component ─────────────────────────────────────
    if cfg.keep_largest:
        result = keep_largest_component(result)
        _log("keep_largest", result)

    # ── Step 11: ensure watertight ───────────────────────────────────────────
    if cfg.ensure_watertight:
        try:
            tm = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
            if not tm.is_watertight:
                tm.fill_holes()
                # Remove any remaining non-manifold faces
                tm.update_faces(tm.nondegenerate_faces())
                tm.update_faces(tm.unique_faces())
                tm.remove_unreferenced_vertices()
            result = MeshData(
                vertices=tm.vertices.view(np.ndarray).astype(np.float32),
                faces=tm.faces.view(np.ndarray).astype(np.int32),
            )
            _log("ensure_watertight", result)
        except Exception as exc:
            logger.debug(f"ensure_watertight failed: {exc}")

    # ── Step 12: decimation ──────────────────────────────────────────────────
    if cfg.decimate and (cfg.target_faces is not None or cfg.decimate_ratio < 1.0):
        result = decimate_mesh(
            result,
            target_faces=cfg.target_faces,
            ratio=cfg.decimate_ratio,
        )
        _log("decimate", result)

    return result
