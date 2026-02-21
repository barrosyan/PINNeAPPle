"""Geometry asset builder registry and build_geometry_asset factory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .geometry import GeometrySpec, GeometryAsset
from .mesh import MeshData


# =========================================================
# Transforms / utilities
# =========================================================
def _rotation_matrix_xyz(rx, ry, rz) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _ensure_vec3(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.shape[0] != 3:
        raise ValueError(f"{name} must be length-3, got shape {a.shape}")
    return a


def _guess_kind_from_path(p: Path) -> str:
    ext = p.suffix.lower().lstrip(".")
    # trimesh: surface mesh formats
    if ext in ("stl", "obj", "ply", "glb", "gltf", "off"):
        return "file"
    # meshio: volumetric / FE formats
    return "mesh_file"


def _maybe_fix_normals(mesh: MeshData, prefer: str = "face") -> None:
    """
    If normals are missing or ill-shaped, compute (numpy-only).
    """
    if mesh.normals is None:
        mesh.ensure_normals(mode=prefer)  # type: ignore[attr-defined]
        return
    # Validate shape (K,3) K in {M,N}
    n = np.asarray(mesh.normals)
    if n.ndim != 2 or n.shape[1] != 3 or n.shape[0] not in (mesh.n_faces, mesh.n_vertices):
        mesh.normals = None
        mesh.ensure_normals(mode=prefer)  # type: ignore[attr-defined]


# =========================================================
# Loaders
# =========================================================
def _load_trimesh_from_file(path: Path):
    import trimesh

    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError("Loaded geometry is not a triangular mesh.")
    if m.vertices is None or m.faces is None:
        raise ValueError("Loaded trimesh has no vertices/faces.")
    return m


def _meshdata_from_trimesh(tm) -> MeshData:
    md = MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None,
    )
    return md


def _meshdata_from_meshio(mesh) -> MeshData:
    # Prefer triangles; if only quads exist, optionally triangulate if options say so (handled later).
    if "triangle" not in mesh.cells_dict:
        raise ValueError("Mesh does not contain triangle cells.")
    faces = mesh.cells_dict["triangle"]
    vertices = mesh.points[:, :3]
    return MeshData(vertices=vertices, faces=faces)


# =========================================================
# Options adapter (keeps backward-compat with getattr)
# =========================================================
def _get_opt(options: Optional[Any], key: str, default: Any = None) -> Any:
    if options is None:
        return default
    return getattr(options, key, default)


def _apply_options(mesh: MeshData, options: Optional[Any]) -> Dict[str, Any]:
    """
    Apply transforms and mesh hygiene steps in a controlled order.

    Returns meta dict of what was applied (safe for manifest).
    """
    meta: Dict[str, Any] = {"applied": []}

    # --- scale ---
    s = _get_opt(options, "scale", None)
    if s is not None:
        s = float(s)
        mesh.vertices *= s
        meta["applied"].append({"op": "scale", "value": s})

    # --- rotate (Euler degrees) ---
    rot_deg = _get_opt(options, "rotate_euler_deg", None)
    if rot_deg is not None:
        rx, ry, rz = rot_deg
        R = _rotation_matrix_xyz(np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz))
        mesh.apply_transform(R, update_normals=True)  # type: ignore[call-arg]
        meta["applied"].append({"op": "rotate_euler_deg", "value": [float(rx), float(ry), float(rz)]})

    # --- translate ---
    t = _get_opt(options, "translate", None)
    if t is not None:
        t3 = _ensure_vec3(t, "translate")
        mesh.vertices += t3[None, :]
        meta["applied"].append({"op": "translate", "value": t3.tolist()})

    # --- normalize / center convenience ---
    if _get_opt(options, "center_to_origin", False):
        c = mesh.vertices.mean(axis=0)
        mesh.vertices -= c[None, :]
        meta["applied"].append({"op": "center_to_origin", "value": c.tolist()})

    if _get_opt(options, "normalize_to_unit_box", False):
        # uses method from your expanded MeshData; fallback if absent
        if hasattr(mesh, "normalize_to_unit_box"):
            sc = mesh.normalize_to_unit_box()  # type: ignore[attr-defined]
            meta["applied"].append({"op": "normalize_to_unit_box", "scale": float(sc)})
        else:
            b0, b1 = mesh.bounds()
            size = (b1 - b0)
            max_dim = float(np.max(size))
            sc = 1.0 / max(max_dim, 1e-12)
            mesh.vertices -= (0.5 * (b0 + b1))[None, :]
            mesh.vertices *= sc
            meta["applied"].append({"op": "normalize_to_unit_box", "scale": float(sc)})

    # --- mesh hygiene ---
    if _get_opt(options, "remove_unused_vertices", False) and hasattr(mesh, "remove_unused_vertices"):
        mesh.remove_unused_vertices()  # type: ignore[attr-defined]
        meta["applied"].append({"op": "remove_unused_vertices"})

    # --- normals ---
    if _get_opt(options, "ensure_normals", True):
        prefer = _get_opt(options, "normals_mode", "face")
        _maybe_fix_normals(mesh, prefer=str(prefer))

    return meta


# =========================================================
# Main builders
# =========================================================
def build_geometry_asset(
    spec: Dict[str, Any] | GeometrySpec,
    *,
    options: Optional[Any] = None,
) -> GeometryAsset:
    """
    Build a GeometryAsset from a GeometrySpec or spec dict.

    Supported kinds:
      - file       (STL/OBJ/PLY/GLTF/OFF) via trimesh
      - mesh_file  (VTK/VTU/MSH/...) via meshio
      - primitive  (delegated to pinneaple_geom.gen.primitives)

    Improvements:
      - spec.validate() when available
      - robust meta (fingerprints, transforms applied)
      - optional normalization/centering/hygiene
      - normals handling
      - boundary group helpers passthrough
    """
    if isinstance(spec, dict):
        spec = GeometrySpec(**spec)

    if hasattr(spec, "validate"):
        spec.validate()  # type: ignore[attr-defined]

    kind = spec.kind.lower()

    # ---------- Load mesh ----------
    if kind in ("file", "mesh_file"):
        path = Path(spec.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        if kind == "file":
            tm = _load_trimesh_from_file(path)
            mesh = _meshdata_from_trimesh(tm)

        else:  # mesh_file
            import meshio

            mio = meshio.read(path)
            mesh = _meshdata_from_meshio(mio)

    elif kind == "primitive":
        try:
            from pinneaple_geom.gen.primitives import build_primitive  # type: ignore
        except Exception as e:
            raise ImportError("Primitive generation requires pinneaple_geom.gen.primitives") from e

        mesh = build_primitive(spec.name, **spec.params)
        if not isinstance(mesh, MeshData):
            # allow primitive builders that return trimesh-like or dict-like, but keep it simple:
            raise TypeError("build_primitive must return MeshData")

    else:
        raise ValueError(f"Unsupported GeometrySpec kind: {kind}")

    # ---------- Apply options ----------
    opt_meta = _apply_options(mesh, options)

    # ---------- Build asset ----------
    bmin, bmax = mesh.bounds()

    # provenance / fingerprints (keep it stable and cheap)
    spec_fp = spec.fingerprint() if hasattr(spec, "fingerprint") else None
    mesh_fp = mesh.fingerprint() if hasattr(mesh, "fingerprint") else None

    asset = GeometryAsset(
        mesh=mesh,
        bounds=(bmin, bmax),
        units=_get_opt(options, "units", None),
        boundary_groups=_get_opt(options, "boundary_labels", {}) or {},
        meta={
            "kind": kind,
            "source": spec.path or spec.name,
            "spec_fingerprint": spec_fp,
            "mesh_fingerprint": mesh_fp,
            "options_applied": opt_meta,
        },
    )

    # optional: store boundary groups from bbox planes into the asset
    if _get_opt(options, "auto_bbox_boundary_groups", False) and hasattr(mesh, "boundary_groups_bbox"):
        tol = float(_get_opt(options, "bbox_boundary_tol", 1e-6))
        groups = mesh.boundary_groups_bbox(tol=tol)  # type: ignore[attr-defined]
        # merge (don't overwrite existing)
        for k, v in groups.items():
            asset.boundary_groups.setdefault(k, v)

    return asset


def load_geometry_asset(
    geom: Any,
    *,
    options: Optional[Any] = None,
) -> GeometryAsset:
    """
    Convenience wrapper:
      - Path / str -> infer spec
      - GeometryAsset -> returned as-is
      - dict -> GeometrySpec

    Improvements:
      - supports already-built GeometrySpec
      - better kind inference
      - keeps backward behavior
    """
    if isinstance(geom, GeometryAsset):
        return geom

    if isinstance(geom, GeometrySpec):
        return build_geometry_asset(geom, options=options)

    if isinstance(geom, (str, Path)):
        p = Path(geom).expanduser().resolve()
        kind = _guess_kind_from_path(p)
        spec = GeometrySpec(kind=kind, path=str(p))
        return build_geometry_asset(spec, options=options)

    if isinstance(geom, dict):
        return build_geometry_asset(geom, options=options)

    raise TypeError(f"Unsupported geometry input type: {type(geom)}")