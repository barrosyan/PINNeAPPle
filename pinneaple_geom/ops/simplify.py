"""Mesh simplification via trimesh or Open3D quadric decimation."""
from __future__ import annotations

from typing import Optional
import numpy as np

from pinneaple_geom.core.mesh import MeshData


def _to_trimesh(mesh: MeshData):
    import trimesh
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )


def _from_trimesh(tm) -> MeshData:
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=(tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None),
    )


def _open3d_available() -> bool:
    try:
        import open3d as o3d  # noqa: F401
        return True
    except Exception:
        return False


def _simplify_open3d(mesh: MeshData, target_faces: int) -> MeshData:
    import open3d as o3d

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    om = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(v),
        triangles=o3d.utility.Vector3iVector(f),
    )
    # Open3D expects target number of triangles
    om2 = om.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    om2.remove_duplicated_vertices()
    om2.remove_degenerate_triangles()
    om2.remove_duplicated_triangles()
    om2.remove_non_manifold_edges()

    v2 = np.asarray(om2.vertices, dtype=np.float64)
    f2 = np.asarray(om2.triangles, dtype=np.int64)

    # no face normals provided by default here
    return MeshData(vertices=v2, faces=f2, normals=None)


def simplify_mesh(
    mesh: MeshData,
    *,
    target_faces: int = 10_000,
    backend: str = "auto",
    on_missing: str = "error",
) -> MeshData:
    """
    Simplify a mesh.

    backend:
      - "auto": try trimesh fast_simplification, else open3d, else fallback
      - "trimesh": only trimesh (requires fast_simplification)
      - "open3d": only open3d

    on_missing:
      - "error": raise if no backend available
      - "return_input": return the input mesh unchanged
    """
    tf = int(target_faces)
    if tf <= 0:
        raise ValueError("target_faces must be > 0")

    backend = (backend or "auto").lower().strip()
    on_missing = (on_missing or "error").lower().strip()

    # -------------------------
    # 1) trimesh path
    # -------------------------
    def _try_trimesh() -> Optional[MeshData]:
        try:
            import trimesh  # noqa
            tm = _to_trimesh(mesh)

            # current number of faces
            n_faces = int(len(tm.faces))
            tf_local = int(tf)

            if tf_local >= n_faces:
                return _from_trimesh(tm)

            # 1) Prefer explicit target_count if supported by fast_simplification wrapper
            try:
                tm2 = tm.simplify_quadric_decimation(face_count=tf_local)
                return _from_trimesh(tm2)
            except TypeError:
                pass  # signature doesn't accept target_count
            except ValueError:
                pass  # wrong arg interpreted

            # 2) Some variants expect a reduction fraction in [0,1]
            # target_reduction = fraction of faces to remove
            target_reduction = float(max(0.0, min(1.0, 1.0 - (tf_local / max(n_faces, 1)))))

            try:
                tm2 = tm.simplify_quadric_decimation(percent=target_reduction)
                return _from_trimesh(tm2)
            except TypeError:
                pass
            except ValueError:
                pass

            # 3) As a final attempt, call with a single positional arg interpreted as reduction
            # (only if it is in [0,1])
            if 0.0 <= target_reduction <= 1.0:
                tm2 = tm.simplify_quadric_decimation(target_reduction)
                return _from_trimesh(tm2)

            return None

        except ModuleNotFoundError:
            return None
        except Exception:
            return None

    # -------------------------
    # 2) open3d path
    # -------------------------
    def _try_open3d() -> Optional[MeshData]:
        if not _open3d_available():
            return None
        try:
            return _simplify_open3d(mesh, tf)
        except Exception:
            return None

    # Dispatch
    if backend == "trimesh":
        out = _try_trimesh()
        if out is not None:
            return out
        if on_missing == "return_input":
            return mesh
        raise RuntimeError(
            "Mesh simplification backend 'trimesh' failed or is missing dependencies. "
            "Install 'fast_simplification' or use backend='open3d'."
        )

    if backend == "open3d":
        out = _try_open3d()
        if out is not None:
            return out
        if on_missing == "return_input":
            return mesh
        raise RuntimeError(
            "Mesh simplification backend 'open3d' is not available or failed. "
            "Install open3d or use backend='trimesh' with fast_simplification."
        )

    # auto
    out = _try_trimesh()
    if out is not None:
        return out

    out = _try_open3d()
    if out is not None:
        return out

    if on_missing == "return_input":
        return mesh

    raise RuntimeError(
        "Mesh simplification failed: no available backend.\n"
        "Fix options:\n"
        "  - pip install fast_simplification   (enables trimesh quadric decimation)\n"
        "  - pip install open3d                (fallback decimator)\n"
        "Or call simplify_mesh(..., on_missing='return_input') to skip."
    )
