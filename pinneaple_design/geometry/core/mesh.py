"""MeshData container for triangle meshes with PINN sampling and collocation support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import hashlib


InsideFn = Callable[[np.ndarray], np.ndarray]  # (N,3)->(N,) bool
SdfFn = Callable[[np.ndarray], np.ndarray]     # (N,3)->(N,) float (negative inside)


@dataclass
class MeshData:
    """
    Lightweight mesh container (numpy-only).

    vertices: (N,3) float64
    faces:    (M,3) int64 (triangle mesh)
    normals:  (M,3) or (N,3), optional

    PINN-ready additions:
      - collocation samplers:
          * sample_surface(n)
          * sample_interior(n, inside_fn=..., sdf_fn=..., domain="mesh|bbox")
          * sample_collocation(...)
      - boundary groups from bbox planes (inlet/outlet/wall-like):
          * boundary_groups_bbox(...)
      - Monte Carlo integration helpers:
          * mc_integral_weights(...)
      - batching utilities:
          * iter_batches(...)
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None

    # =====================================================
    # Init / validation
    # =====================================================
    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float64)
        self._validate()

    def _validate(self):
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be shape (N,3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be shape (M,3)")
        if np.any(self.faces < 0) or np.any(self.faces >= self.vertices.shape[0]):
            raise ValueError("faces contain invalid vertex indices")
        if self.normals is not None:
            if self.normals.ndim != 2 or self.normals.shape[1] != 3:
                raise ValueError("normals must be shape (K,3)")
            if self.normals.shape[0] not in (self.n_faces, self.n_vertices):
                raise ValueError("normals must be (M,3) (face) or (N,3) (vertex)")

    # =====================================================
    # Basic properties
    # =====================================================
    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def bbox_size(self) -> np.ndarray:
        b0, b1 = self.bounds()
        return b1 - b0

    def center(self) -> np.ndarray:
        b0, b1 = self.bounds()
        return 0.5 * (b0 + b1)

    # =====================================================
    # Core transforms
    # =====================================================
    def apply_transform(self, matrix: np.ndarray, update_normals: bool = True) -> None:
        """
        Apply 4x4 homogeneous transform in-place.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")

        v = np.ones((self.vertices.shape[0], 4), dtype=np.float64)
        v[:, :3] = self.vertices
        vt = (matrix @ v.T).T
        self.vertices = vt[:, :3]

        if update_normals and self.normals is not None:
            R = matrix[:3, :3]
            self.normals = (R @ self.normals.T).T
            self._normalize_normals()

    def translate(self, delta: np.ndarray) -> None:
        self.vertices += np.asarray(delta, dtype=np.float64)

    def scale(self, factor: float, about: Optional[np.ndarray] = None) -> None:
        factor = float(factor)
        if about is None:
            about = np.zeros(3, dtype=np.float64)
        about = np.asarray(about, dtype=np.float64)
        self.vertices = (self.vertices - about) * factor + about

    def normalize_to_unit_box(self) -> float:
        """
        Center to origin and scale so largest bbox dimension = 1.
        Returns applied scale.
        """
        size = self.bbox_size()
        max_dim = float(np.max(size))
        s = 1.0 / max(max_dim, 1e-12)
        self.translate(-self.center())
        self.scale(s, about=np.zeros(3))
        return s

    # =====================================================
    # Normals
    # =====================================================
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-12)

    def _normalize_normals(self) -> None:
        self.normals = self._normalize(self.normals)

    def compute_face_normals(self) -> np.ndarray:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        return self._normalize(n)

    def compute_vertex_normals(self) -> np.ndarray:
        fn = self.compute_face_normals()
        vn = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(vn, self.faces[:, i], fn)
        return self._normalize(vn)

    def ensure_normals(self, mode: str = "vertex") -> None:
        if self.normals is not None:
            return
        if mode == "face":
            self.normals = self.compute_face_normals()
        else:
            self.normals = self.compute_vertex_normals()

    # =====================================================
    # Geometry metrics (useful for weighting losses)
    # =====================================================
    def face_areas(self) -> np.ndarray:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    def surface_area(self) -> float:
        return float(np.sum(self.face_areas()))

    def volume_bbox(self) -> float:
        return float(np.prod(self.bbox_size()))

    # =====================================================
    # Sampling: surface (tri-area uniform)
    # =====================================================
    def sample_surface(self, n: int, return_face_ids: bool = False) -> np.ndarray:
        """
        Uniform random sampling over surface area.
        Returns points (N,3). Optionally also returns face_ids (N,).
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")

        areas = self.face_areas()
        s = float(np.sum(areas))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Mesh has zero/invalid surface area")

        probs = areas / s
        face_ids = np.random.choice(self.n_faces, size=n, p=probs)

        v0 = self.vertices[self.faces[face_ids, 0]]
        v1 = self.vertices[self.faces[face_ids, 1]]
        v2 = self.vertices[self.faces[face_ids, 2]]

        # barycentric sampling
        u = np.random.rand(n, 1)
        v = np.random.rand(n, 1)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]

        pts = v0 + u * (v1 - v0) + v * (v2 - v0)
        if return_face_ids:
            return pts, face_ids
        return pts

    # =====================================================
    # Sampling: interior (PINN collocation)
    # =====================================================
    def sample_bbox(self, n: int, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Sample uniformly inside a bounding box (default: mesh bbox).
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        if bounds is None:
            b0, b1 = self.bounds()
        else:
            b0, b1 = bounds
            b0 = np.asarray(b0, dtype=np.float64)
            b1 = np.asarray(b1, dtype=np.float64)
        r = np.random.rand(n, 3)
        return b0 + r * (b1 - b0)

    def sample_interior(
        self,
        n: int,
        *,
        domain: str = "bbox",
        inside_fn: Optional[InsideFn] = None,
        sdf_fn: Optional[SdfFn] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        max_tries: int = 50,
    ) -> np.ndarray:
        """
        Sample collocation points in the interior domain.

        domain:
          - "bbox": interior is the bounding box (works without any extra info)
          - "mesh": requires inside_fn or sdf_fn (numpy-only) to test inside

        Prefer sdf_fn if you have it (negative inside). Otherwise inside_fn boolean.

        If domain="mesh" and no inside/sdf is given, it falls back to bbox.
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")

        if bounds is None:
            b0, b1 = self.bounds()
        else:
            b0, b1 = bounds

        if domain not in ("bbox", "mesh"):
            raise ValueError("domain must be 'bbox' or 'mesh'")

        if domain == "bbox" or (inside_fn is None and sdf_fn is None):
            return self.sample_bbox(n, bounds=(b0, b1))

        # rejection sampling in bbox using inside predicate
        out = []
        remaining = n
        tries = 0

        while remaining > 0 and tries < int(max_tries):
            tries += 1
            # oversample to improve acceptance
            m = int(max(remaining * 2, 256))
            cand = self.sample_bbox(m, bounds=(b0, b1))

            if sdf_fn is not None:
                mask = np.asarray(sdf_fn(cand)) < 0.0
            else:
                mask = np.asarray(inside_fn(cand), dtype=bool)

            good = cand[mask]
            if good.size == 0:
                continue

            take = min(remaining, good.shape[0])
            out.append(good[:take])
            remaining -= take

        if remaining > 0:
            # best-effort: fill the rest with bbox points (keeps pipeline running)
            out.append(self.sample_bbox(remaining, bounds=(b0, b1)))

        return np.vstack(out)

    # =====================================================
    # Boundary groups (bbox-plane heuristic)
    # =====================================================
    def boundary_groups_bbox(
        self,
        tol: float = 1e-6,
        *,
        names: Tuple[str, str, str, str, str, str] = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Build simple boundary groups using bbox planes.

        Returns dict like:
          {
            "xmin": {"vertex_ids": ..., "mask": ...},
            ...
          }

        Useful when your PDE domain is box-like or you need quick inlet/outlet/walls.
        """
        tol = float(tol)
        b0, b1 = self.bounds()
        x0, y0, z0 = b0
        x1, y1, z1 = b1

        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.vertices[:, 2]

        masks = [
            np.abs(x - x0) <= tol,
            np.abs(x - x1) <= tol,
            np.abs(y - y0) <= tol,
            np.abs(y - y1) <= tol,
            np.abs(z - z0) <= tol,
            np.abs(z - z1) <= tol,
        ]

        out: Dict[str, Dict[str, np.ndarray]] = {}
        for nm, m in zip(names, masks):
            ids = np.nonzero(m)[0].astype(np.int64)
            out[nm] = {"vertex_ids": ids, "mask": m}
        return out

    # =====================================================
    # Collocation pack (PINN training input)
    # =====================================================
    def sample_collocation(
        self,
        n_interior: int,
        n_boundary: int,
        *,
        interior_domain: str = "bbox",
        inside_fn: Optional[InsideFn] = None,
        sdf_fn: Optional[SdfFn] = None,
        boundary_mode: str = "surface",
        boundary_groups: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dict ready for PINN batching.

        boundary_mode:
          - "surface": boundary points sampled from mesh surface (triangle sampling)
          - "bbox_planes": boundary points sampled on bbox planes (fast box BCs)

        If boundary_groups is provided, will also return:
          - "bnd_group_id" (N,) int64, mapping each boundary point to group index
          - "bnd_group_name" as an array of fixed-length strings (optional convenience)
        """
        n_interior = int(n_interior)
        n_boundary = int(n_boundary)
        if n_interior < 0 or n_boundary < 0:
            raise ValueError("n_interior and n_boundary must be >= 0")

        if bounds is None:
            b0, b1 = self.bounds()
        else:
            b0, b1 = bounds

        X_int = (
            self.sample_interior(
                n_interior,
                domain=interior_domain,
                inside_fn=inside_fn,
                sdf_fn=sdf_fn,
                bounds=(b0, b1),
            )
            if n_interior > 0
            else np.zeros((0, 3), dtype=np.float64)
        )

        if n_boundary == 0:
            X_bnd = np.zeros((0, 3), dtype=np.float64)
            return {"X_int": X_int, "X_bnd": X_bnd}

        if boundary_mode == "surface":
            X_bnd = self.sample_surface(n_boundary, return_face_ids=False)

        elif boundary_mode == "bbox_planes":
            # sample points uniformly over the 6 faces of the bbox
            b0 = np.asarray(b0, dtype=np.float64)
            b1 = np.asarray(b1, dtype=np.float64)
            areas = np.array(
                [
                    (b1[1] - b0[1]) * (b1[2] - b0[2]),  # xmin
                    (b1[1] - b0[1]) * (b1[2] - b0[2]),  # xmax
                    (b1[0] - b0[0]) * (b1[2] - b0[2]),  # ymin
                    (b1[0] - b0[0]) * (b1[2] - b0[2]),  # ymax
                    (b1[0] - b0[0]) * (b1[1] - b0[1]),  # zmin
                    (b1[0] - b0[0]) * (b1[1] - b0[1]),  # zmax
                ],
                dtype=np.float64,
            )
            probs = areas / np.sum(areas)
            face_id = np.random.choice(6, size=n_boundary, p=probs)

            r = np.random.rand(n_boundary, 2)
            X_bnd = np.zeros((n_boundary, 3), dtype=np.float64)

            # maps: each case sets one coord fixed and two coords random
            # xmin/xmax => (y,z) random
            m = face_id == 0
            X_bnd[m, 0] = b0[0]
            X_bnd[m, 1] = b0[1] + r[m, 0] * (b1[1] - b0[1])
            X_bnd[m, 2] = b0[2] + r[m, 1] * (b1[2] - b0[2])

            m = face_id == 1
            X_bnd[m, 0] = b1[0]
            X_bnd[m, 1] = b0[1] + r[m, 0] * (b1[1] - b0[1])
            X_bnd[m, 2] = b0[2] + r[m, 1] * (b1[2] - b0[2])

            # ymin/ymax => (x,z) random
            m = face_id == 2
            X_bnd[m, 1] = b0[1]
            X_bnd[m, 0] = b0[0] + r[m, 0] * (b1[0] - b0[0])
            X_bnd[m, 2] = b0[2] + r[m, 1] * (b1[2] - b0[2])

            m = face_id == 3
            X_bnd[m, 1] = b1[1]
            X_bnd[m, 0] = b0[0] + r[m, 0] * (b1[0] - b0[0])
            X_bnd[m, 2] = b0[2] + r[m, 1] * (b1[2] - b0[2])

            # zmin/zmax => (x,y) random
            m = face_id == 4
            X_bnd[m, 2] = b0[2]
            X_bnd[m, 0] = b0[0] + r[m, 0] * (b1[0] - b0[0])
            X_bnd[m, 1] = b0[1] + r[m, 1] * (b1[1] - b0[1])

            m = face_id == 5
            X_bnd[m, 2] = b1[2]
            X_bnd[m, 0] = b0[0] + r[m, 0] * (b1[0] - b0[0])
            X_bnd[m, 1] = b0[1] + r[m, 1] * (b1[1] - b0[1])

        else:
            raise ValueError("boundary_mode must be 'surface' or 'bbox_planes'")

        pack: Dict[str, np.ndarray] = {"X_int": X_int, "X_bnd": X_bnd}

        # Optional: attach group ids for boundary points
        if boundary_groups is not None and n_boundary > 0:
            # only supports vertex-defined bbox plane groups directly;
            # for "surface" points, group assignment would need SDF/nearest projection.
            # Here we support bbox_planes mode by using the sampled face_id if user passes matching names.
            if boundary_mode == "bbox_planes":
                group_names = list(boundary_groups.keys())
                # Try to map common naming -> sampled face ordering
                # If keys match ("xmin","xmax","ymin","ymax","zmin","zmax"), this becomes perfect.
                default_order = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                name_to_gid = {nm: i for i, nm in enumerate(group_names)}
                gid = np.full((n_boundary,), -1, dtype=np.int64)
                for fid, nm in enumerate(default_order):
                    if nm in name_to_gid:
                        gid[face_id == fid] = name_to_gid[nm]
                pack["bnd_group_id"] = gid
                # fixed-length unicode array for convenience
                maxlen = max((len(nm) for nm in group_names), default=1)
                bnames = np.full((len(group_names),), "", dtype=f"<U{maxlen}")
                for i, nm in enumerate(group_names):
                    bnames[i] = nm
                pack["bnd_group_name_lut"] = bnames

        return pack

    # =====================================================
    # Monte Carlo integration weights (for loss weighting)
    # =====================================================
    def mc_integral_weights(
        self,
        X: np.ndarray,
        *,
        domain_volume: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns per-sample weights for MC integral approximation:
          ∫ f(x) dx ≈ sum_i w_i f(x_i)

        If mask is provided, weights apply only to masked points (others 0).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("X must be (N,3)")

        N = X.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=np.float64)

        V = float(domain_volume) if domain_volume is not None else self.volume_bbox()
        w = np.full((N,), V / max(N, 1), dtype=np.float64)

        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (N,):
                raise ValueError("mask must be shape (N,)")
            w = np.where(m, w, 0.0)

        return w

    # =====================================================
    # Batching helper (PINN training loops)
    # =====================================================
    def iter_batches(
        self,
        X: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        Yield mini-batches of X (N,D). Numpy-only.
        """
        X = np.asarray(X)
        N = X.shape[0]
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError("batch_size must be > 0")

        idx = np.arange(N)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)

        for i in range(0, N, bs):
            yield X[idx[i : i + bs]]

    # =====================================================
    # Hashing / caching
    # =====================================================
    def fingerprint(self) -> str:
        payload = (
            str(self.vertices.shape).encode("utf-8") +
            str(self.vertices.dtype).encode("utf-8") +
            self.vertices.tobytes(order="C") +
            str(self.faces.shape).encode("utf-8") +
            str(self.faces.dtype).encode("utf-8") +
            self.faces.tobytes(order="C")
        )
        return hashlib.sha256(payload).hexdigest()

    # =====================================================
    # Copy
    # =====================================================
    def copy(self) -> "MeshData":
        return MeshData(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=None if self.normals is None else self.normals.copy(),
        )