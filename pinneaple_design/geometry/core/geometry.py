"""GeometrySpec, GeometryAsset, and mesh transform utilities for Pinneaple geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import hashlib
import json

import numpy as np


# =========================================================
# Helpers (no external deps)
# =========================================================
def _as_np3(x: Any, name: str) -> np.ndarray:
    """Convert input to (3,) float64 array; raise if shape is not (3,)."""
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError(f"{name} must be shape (3,), got {a.shape}")
    return a


def _stable_json(obj: Any) -> str:
    """Serialize object to canonical JSON string for fingerprinting."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_bytes(data: bytes, algo: str = "sha256") -> str:
    """Compute hex digest of raw bytes using the given algorithm."""
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def _hash_array(a: np.ndarray, algo: str = "sha256") -> str:
    """Compute stable hash of array (shape + dtype + bytes) for fingerprinting."""
    a = np.asarray(a)
    # Include shape + dtype + raw bytes for robust fingerprinting
    payload = (
        str(a.shape).encode("utf-8")
        + b"|"
        + str(a.dtype).encode("utf-8")
        + b"|"
        + a.tobytes(order="C")
    )
    return _hash_bytes(payload, algo=algo)


def _compute_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) for (N,3) vertex array."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {v.shape}")
    return v.min(axis=0), v.max(axis=0)


def _apply_transform(vertices: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 homogeneous transform to (N,3) vertices.
    """
    v = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be (4,4), got {T.shape}")
    ones = np.ones((v.shape[0], 1), dtype=np.float64)
    vh = np.concatenate([v, ones], axis=1)  # (N,4)
    out = (vh @ T.T)[:, :3]
    return out


def _normalize_vecs(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows of (N,3) array to unit length."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-face normals for triangular faces (M,3 int indices).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (M,3), got {f.shape}")
    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]
    n = np.cross(b - a, c - a)
    return _normalize_vecs(n)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Simple angle-unaware vertex normals: accumulate adjacent face normals.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = _face_normals(v, f)
    vn = np.zeros_like(v)
    # Accumulate
    for i in range(3):
        np.add.at(vn, f[:, i], fn)
    return _normalize_vecs(vn)


# =========================================================
# Geometry Spec
# =========================================================
@dataclass
class GeometrySpec:
    """
    Declarative geometry specification.

    Examples:
      {"kind":"primitive","name":"box","params":{...}}
      {"kind":"file","path":"model.stl"}
      {"kind":"mesh_file","path":"case.vtu"}

    Improvements:
      - validation (basic)
      - stable fingerprinting for caching/manifests
      - small convenience helpers
    """
    kind: str
    name: Optional[str] = None
    path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("GeometrySpec.kind must be a non-empty string")

        if self.kind in ("file", "mesh_file"):
            if not self.path:
                raise ValueError(f"GeometrySpec(kind={self.kind}) requires 'path'")
        if self.kind == "primitive":
            if not self.name:
                raise ValueError("GeometrySpec(kind='primitive') requires 'name'")

        if not isinstance(self.params, dict):
            raise ValueError("GeometrySpec.params must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "params": self.params,
        }

    def fingerprint(self, algo: str = "sha256") -> str:
        """
        Stable hash of the declarative spec, useful for cache keys.
        """
        self.validate()
        payload = _stable_json(self.to_dict()).encode("utf-8")
        return _hash_bytes(payload, algo=algo)

    def is_file_backed(self) -> bool:
        return self.kind in ("file", "mesh_file") and bool(self.path)


# =========================================================
# Geometry Asset
# =========================================================
@dataclass
class GeometryAsset:
    """
    Unified geometry container used across Pinneaple.

    Holds a MeshData + metadata and optional boundary groups.

    Attributes:
      - mesh: MeshData (expects .vertices (N,3), .faces (M,3))
      - bounds: (min_xyz, max_xyz)
      - units: optional physical units (e.g. meters)
      - boundary_groups: semantic labels (inlet/outlet/wall/etc.)
      - meta: free metadata (source, transforms applied, hashes)

    Improvements:
      - recompute bounds / normalization
      - transforms (translate/scale/affine 4x4)
      - lightweight geometry fingerprints (verts/faces)
      - optional normal computation (face/vertex)
      - boundary group convenience helpers
      - safe "manifest" summary for storage (e.g., Zarr index)
    """
    mesh: Any  # MeshData-like: has vertices, faces; may optionally have normals
    bounds: Tuple[np.ndarray, np.ndarray]
    units: Optional[str] = None
    boundary_groups: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -------------------------
    # Basic geometry helpers
    # -------------------------
    def bbox_size(self) -> np.ndarray:
        return self.bounds[1] - self.bounds[0]

    def center(self) -> np.ndarray:
        return 0.5 * (self.bounds[0] + self.bounds[1])

    def recompute_bounds(self) -> None:
        self.bounds = _compute_bounds(self.mesh.vertices)

    def ensure_bounds(self) -> None:
        """
        If bounds look invalid, recompute. (Useful after manual mesh edits.)
        """
        b0, b1 = self.bounds
        if b0 is None or b1 is None:
            self.recompute_bounds()
            return
        b0 = np.asarray(b0, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        if b0.shape != (3,) or b1.shape != (3,) or np.any(b1 < b0):
            self.recompute_bounds()

    # -------------------------
    # Fingerprints / caching
    # -------------------------
    def mesh_fingerprint(self, algo: str = "sha256") -> str:
        """
        Hash based on vertices + faces. Great for caching pipelines.
        """
        v = np.asarray(self.mesh.vertices)
        f = np.asarray(self.mesh.faces)
        hv = _hash_array(v, algo=algo)
        hf = _hash_array(f, algo=algo)
        return _hash_bytes(f"{hv}|{hf}".encode("utf-8"), algo=algo)

    def record_fingerprint(self, key: str = "mesh_fingerprint", algo: str = "sha256") -> str:
        fp = self.mesh_fingerprint(algo=algo)
        self.meta[key] = fp
        return fp

    # -------------------------
    # Transforms
    # -------------------------
    def apply_transform(self, T: np.ndarray, record: bool = True) -> None:
        """
        Apply 4x4 transform to vertices (and recompute bounds).
        """
        self.mesh.vertices = _apply_transform(self.mesh.vertices, T)
        self.recompute_bounds()

        if record:
            self.meta.setdefault("transforms", [])
            self.meta["transforms"].append(
                {
                    "type": "affine_4x4",
                    "matrix": np.asarray(T, dtype=np.float64).tolist(),
                }
            )

    def translate(self, delta_xyz: Any, record: bool = True) -> None:
        d = _as_np3(delta_xyz, "delta_xyz")
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = d
        self.apply_transform(T, record=record)

    def scale(self, s: float, about: Optional[Any] = None, record: bool = True) -> None:
        """
        Uniform scale by s, optionally about a point (default: origin).
        """
        s = float(s)
        c = _as_np3(about, "about") if about is not None else np.zeros(3, dtype=np.float64)

        # T = Translate(c) * Scale(s) * Translate(-c)
        T1 = np.eye(4, dtype=np.float64)
        T1[:3, 3] = -c
        S = np.eye(4, dtype=np.float64)
        S[0, 0] = S[1, 1] = S[2, 2] = s
        T2 = np.eye(4, dtype=np.float64)
        T2[:3, 3] = c
        self.apply_transform(T2 @ S @ T1, record=record)

    def normalize_to_unit_box(
        self,
        target_size: float = 1.0,
        center_to_origin: bool = True,
        eps: float = 1e-12,
        record: bool = True,
    ) -> Dict[str, Any]:
        """
        Normalize geometry to fit inside a cube of edge 'target_size'.
        Useful for stable training + batching.

        Returns info dict with scale and original bounds.
        """
        self.ensure_bounds()
        b0, b1 = self.bounds
        size = b1 - b0
        max_dim = float(np.max(size))
        scale = (float(target_size) / max(max_dim, eps))

        info = {
            "original_bounds": (b0.copy(), b1.copy()),
            "scale": scale,
            "centered": bool(center_to_origin),
            "target_size": float(target_size),
        }

        if center_to_origin:
            self.translate(-self.center(), record=record)
        self.scale(scale, about=np.zeros(3), record=record)
        return info

    # -------------------------
    # Normals (optional)
    # -------------------------
    def ensure_normals(self, kind: str = "vertex") -> None:
        """
        Compute normals if mesh doesn't have them.

        kind: "face" or "vertex"
        """
        if kind not in ("face", "vertex"):
            raise ValueError("kind must be 'face' or 'vertex'")

        # If the mesh already has normals, keep them.
        if hasattr(self.mesh, "normals") and self.mesh.normals is not None:
            return

        v = np.asarray(self.mesh.vertices)
        f = np.asarray(self.mesh.faces)
        if kind == "face":
            n = _face_normals(v, f)
        else:
            n = _vertex_normals(v, f)

        # Store in a conventional attribute name
        try:
            self.mesh.normals = n
        except Exception:
            # If mesh object is immutable-ish, store in meta
            self.meta["computed_normals"] = n

    # -------------------------
    # Boundary groups helpers
    # -------------------------
    def add_boundary_group(self, name: str, data: Any) -> None:
        """
        Example: group "inlet" -> {face_ids:[...]} or {vertex_ids:[...]} or mask arrays.
        """
        if not name or not isinstance(name, str):
            raise ValueError("boundary group name must be a non-empty string")
        self.boundary_groups[name] = data

    def list_boundary_groups(self) -> List[str]:
        return sorted(list(self.boundary_groups.keys()))

    def get_boundary_group(self, name: str, default: Any = None) -> Any:
        return self.boundary_groups.get(name, default)

    # -------------------------
    # Serialization-friendly summaries
    # -------------------------
    def manifest(self) -> Dict[str, Any]:
        """
        Storage-friendly metadata (no huge arrays). Good for Zarr inspector/manifests.
        """
        self.ensure_bounds()
        v = np.asarray(self.mesh.vertices)
        f = np.asarray(self.mesh.faces)

        return {
            "n_vertices": int(v.shape[0]),
            "n_faces": int(f.shape[0]),
            "bounds": (self.bounds[0].tolist(), self.bounds[1].tolist()),
            "units": self.units,
            "boundary_groups": self.list_boundary_groups(),
            "meta": {
                # keep meta small; store only scalar-ish entries by default
                k: self.meta[k]
                for k in self.meta.keys()
                if isinstance(self.meta[k], (str, int, float, bool, type(None), list, dict))
            },
            "mesh_fingerprint": self.mesh_fingerprint(),
        }

    def summary(self) -> Dict[str, Any]:
        """
        Backwards-compatible summary (keeps your original keys)
        + adds a couple extra low-cost fields.
        """
        self.ensure_bounds()
        return {
            "n_vertices": int(self.mesh.vertices.shape[0]),
            "n_faces": int(self.mesh.faces.shape[0]),
            "bounds": (self.bounds[0].tolist(), self.bounds[1].tolist()),
            "units": self.units,
            "boundary_groups": list(self.boundary_groups.keys()),
            "meta_keys": list(self.meta.keys()),
            # extras (cheap, useful)
            "bbox_size": self.bbox_size().tolist(),
            "center": self.center().tolist(),
        }

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        units: Optional[str] = None,
        boundary_groups: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        record_fp: bool = True,
    ) -> "GeometryAsset":
        bounds = _compute_bounds(mesh.vertices)
        obj = cls(
            mesh=mesh,
            bounds=bounds,
            units=units,
            boundary_groups=boundary_groups or {},
            meta=meta or {},
        )
        if record_fp:
            obj.record_fingerprint()
        return obj