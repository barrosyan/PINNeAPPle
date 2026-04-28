"""MeshCollocator — 3D collocation point sampler for PINN training.

Bridges the gap between 3D geometry (MeshData from step_to_mesh / STL import)
and PINN training by producing interior collocation points (for PDE residuals)
and surface boundary points (for BC enforcement).

Supports:
- Bounding-box interior sampling (fast, no extra deps)
- Convex-hull interior sampling via scipy (good for convex bodies)
- Trimesh ray-casting for watertight meshes (precise, requires trimesh)
- Hollow geometry: outer mesh minus inner mesh (e.g. pipe, heat sink)

Quick start
-----------
>>> from pinneaple_geom.mesh_collocator import MeshCollocator, MeshCollocatorConfig
>>> from pinneaple_geom.io.step import step_to_mesh, StepImportConfig
>>> mesh = step_to_mesh("part.step", cfg=StepImportConfig(kind="surface"))
>>> col = MeshCollocator(mesh)
>>> batch = col.sample()
>>> batch.X_interior.shape   # (5000, 3)
>>> batch.X_boundary.shape   # (2000, 3)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from pinneaple_geom.core.mesh import MeshData


# ──────────────────────────────────────────────────────────────────────────────
# Config & output types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MeshCollocatorConfig:
    """Configuration for MeshCollocator.

    Parameters
    ----------
    n_interior : int
        Number of interior collocation points to sample.
    n_boundary : int
        Number of surface / boundary collocation points.
    domain : str
        Interior sampling strategy:
        - ``"bbox"`` — uniform inside bounding box (fast, coarse).
        - ``"convex_hull"`` — rejection sampling with scipy ConvexHull test.
        - ``"trimesh"`` — ray-cast inside test (requires *trimesh*; precise).
    boundary_mode : str
        - ``"surface"`` — sample uniformly on triangle surface (area-weighted).
        - ``"bbox_planes"`` — sample on the 6 bounding-box faces.
    hollow : bool
        If True, sample the *fluid* domain: inside the outer mesh but outside
        the inner mesh.  Pass ``inner_mesh`` to :class:`MeshCollocator`.
    seed : int, optional
        Random seed for reproducibility.
    max_rejection_tries : int
        Maximum rejection-sampling rounds before giving up and falling back to
        bbox.  Each round oversamples by 2×.
    """
    n_interior: int = 5000
    n_boundary: int = 2000
    domain: str = "convex_hull"       # "bbox" | "convex_hull" | "trimesh"
    boundary_mode: str = "surface"    # "surface" | "bbox_planes"
    hollow: bool = False
    seed: Optional[int] = None
    max_rejection_tries: int = 100


@dataclass
class CollocationBatch3D:
    """Collocation batch ready for PINN training.

    Attributes
    ----------
    X_interior : np.ndarray, shape (N_int, 3)
        Interior collocation points for PDE residual evaluation.
    X_boundary : np.ndarray, shape (N_bnd, 3)
        Boundary collocation points for BC enforcement.
    normals_boundary : np.ndarray or None, shape (N_bnd, 3)
        Outward surface normals at boundary points (if available).
    boundary_groups : dict or None
        Named boundary groups mapping name → indices into ``X_boundary``.
        Populated when ``boundary_mode="bbox_planes"``.
    """
    X_interior: np.ndarray
    X_boundary: np.ndarray
    normals_boundary: Optional[np.ndarray] = None
    boundary_groups: Optional[Dict[str, np.ndarray]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class MeshCollocator:
    """Collocation point sampler for 3D mesh geometry.

    Parameters
    ----------
    mesh : MeshData
        Outer (or only) surface/volume mesh.
    config : MeshCollocatorConfig, optional
    inner_mesh : MeshData, optional
        Inner surface mesh for hollow-geometry setups (e.g. the solid core
        inside a fluid domain).  Only used when ``config.hollow=True``.
    """

    def __init__(
        self,
        mesh: MeshData,
        config: Optional[MeshCollocatorConfig] = None,
        *,
        inner_mesh: Optional[MeshData] = None,
    ):
        self.mesh = mesh
        self.inner_mesh = inner_mesh
        self.cfg = config or MeshCollocatorConfig()

        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

        self._hull = None
        self._inner_hull = None
        self._trimesh_obj = None
        self._inner_trimesh_obj = None

    # ------------------------------------------------------------------
    # Lazy helpers — built on first use
    # ------------------------------------------------------------------

    def _get_hull(self):
        if self._hull is None:
            from scipy.spatial import ConvexHull  # type: ignore
            self._hull = ConvexHull(self.mesh.vertices)
        return self._hull

    def _get_inner_hull(self):
        if self._inner_hull is None and self.inner_mesh is not None:
            from scipy.spatial import ConvexHull  # type: ignore
            self._inner_hull = ConvexHull(self.inner_mesh.vertices)
        return self._inner_hull

    def _get_trimesh(self):
        if self._trimesh_obj is None:
            import trimesh  # type: ignore
            self._trimesh_obj = trimesh.Trimesh(
                vertices=self.mesh.vertices,
                faces=self.mesh.faces,
                process=False,
            )
        return self._trimesh_obj

    def _get_inner_trimesh(self):
        if self._inner_trimesh_obj is None and self.inner_mesh is not None:
            import trimesh  # type: ignore
            self._inner_trimesh_obj = trimesh.Trimesh(
                vertices=self.inner_mesh.vertices,
                faces=self.inner_mesh.faces,
                process=False,
            )
        return self._inner_trimesh_obj

    # ------------------------------------------------------------------
    # Inside-test predicates
    # ------------------------------------------------------------------

    def _inside_hull(self, pts: np.ndarray, hull) -> np.ndarray:
        """Test containment using half-space equations of a ConvexHull."""
        # hull.equations: (nfacets, 4) — [normal | offset]
        # Point p is inside if all: equations[:, :3] @ p + equations[:, 3] <= 0
        eqs = hull.equations  # (F, 4)
        dot = pts @ eqs[:, :3].T + eqs[:, 3]  # (N, F)
        return np.all(dot <= 1e-10, axis=1)

    def _inside_trimesh(self, pts: np.ndarray, tm) -> np.ndarray:
        return tm.contains(pts)

    def _inside_outer(self, pts: np.ndarray) -> np.ndarray:
        domain = self.cfg.domain
        if domain == "bbox":
            b0, b1 = self.mesh.bounds()
            return np.all((pts >= b0) & (pts <= b1), axis=1)
        if domain == "convex_hull":
            return self._inside_hull(pts, self._get_hull())
        if domain == "trimesh":
            return self._inside_trimesh(pts, self._get_trimesh())
        raise ValueError(f"Unknown domain: {domain!r}")

    def _inside_inner(self, pts: np.ndarray) -> np.ndarray:
        """Returns True for points that are INSIDE the inner mesh (to be excluded)."""
        if self.inner_mesh is None:
            return np.zeros(len(pts), dtype=bool)
        domain = self.cfg.domain
        if domain in ("bbox", "convex_hull"):
            return self._inside_hull(pts, self._get_inner_hull())
        if domain == "trimesh":
            return self._inside_trimesh(pts, self._get_inner_trimesh())
        raise ValueError(f"Unknown domain: {domain!r}")

    # ------------------------------------------------------------------
    # Interior sampling
    # ------------------------------------------------------------------

    def sample_interior(self, n: int) -> np.ndarray:
        """Sample ``n`` interior collocation points."""
        cfg = self.cfg
        b0, b1 = self.mesh.bounds()

        if cfg.domain == "bbox" and not cfg.hollow:
            # Fast path: no inside test needed
            r = np.random.rand(n, 3)
            return b0 + r * (b1 - b0)

        # Rejection sampling
        accepted: list = []
        remaining = n
        tries = 0

        while remaining > 0 and tries < cfg.max_rejection_tries:
            tries += 1
            m = max(remaining * 3, 512)
            r = np.random.rand(m, 3)
            cand = b0 + r * (b1 - b0)

            mask_outer = self._inside_outer(cand)
            if cfg.hollow:
                mask_inner = self._inside_inner(cand)
                mask = mask_outer & ~mask_inner
            else:
                mask = mask_outer

            good = cand[mask]
            if good.shape[0] == 0:
                continue

            take = min(remaining, good.shape[0])
            accepted.append(good[:take])
            remaining -= take

        if remaining > 0:
            # Fallback: fill with bbox points so training can proceed
            r = np.random.rand(remaining, 3)
            accepted.append(b0 + r * (b1 - b0))

        return np.vstack(accepted)

    # ------------------------------------------------------------------
    # Boundary sampling
    # ------------------------------------------------------------------

    def sample_boundary(self, n: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Sample ``n`` boundary points and (optionally) their normals.

        Returns (X_bnd, normals) where normals may be None.
        """
        mode = self.cfg.boundary_mode

        if mode == "surface":
            pts, face_ids = self.mesh.sample_surface(n, return_face_ids=True)
            # Compute normals at the sampled faces
            fn = self.mesh.compute_face_normals()
            normals = fn[face_ids]
            return pts, normals

        if mode == "bbox_planes":
            b0, b1 = self.mesh.bounds()
            pts = np.zeros((n, 3), dtype=np.float64)
            normals = np.zeros((n, 3), dtype=np.float64)
            # Area of each face
            dims = b1 - b0
            areas = np.array([
                dims[1] * dims[2],  # xmin / xmax
                dims[1] * dims[2],
                dims[0] * dims[2],  # ymin / ymax
                dims[0] * dims[2],
                dims[0] * dims[1],  # zmin / zmax
                dims[0] * dims[1],
            ], dtype=np.float64)
            probs = areas / areas.sum()
            face_ids = np.random.choice(6, size=n, p=probs)
            r = np.random.rand(n, 2)

            _normals_lut = np.array([
                [-1, 0, 0], [1, 0, 0],
                [0, -1, 0], [0, 1, 0],
                [0, 0, -1], [0, 0, 1],
            ], dtype=np.float64)
            _fixed_dim = [0, 0, 1, 1, 2, 2]
            _fixed_val = [b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]]
            _free_dims = [(1, 2), (1, 2), (0, 2), (0, 2), (0, 1), (0, 1)]
            _free_lo   = [
                (b0[1], b0[2]), (b0[1], b0[2]),
                (b0[0], b0[2]), (b0[0], b0[2]),
                (b0[0], b0[1]), (b0[0], b0[1]),
            ]
            _free_hi   = [
                (b1[1], b1[2]), (b1[1], b1[2]),
                (b1[0], b1[2]), (b1[0], b1[2]),
                (b1[0], b1[1]), (b1[0], b1[1]),
            ]

            for fid in range(6):
                m = face_ids == fid
                if not m.any():
                    continue
                fd, fv = _fixed_dim[fid], _fixed_val[fid]
                d0, d1 = _free_dims[fid]
                lo0, lo1 = _free_lo[fid]
                hi0, hi1 = _free_hi[fid]
                pts[m, fd] = fv
                pts[m, d0] = lo0 + r[m, 0] * (hi0 - lo0)
                pts[m, d1] = lo1 + r[m, 1] * (hi1 - lo1)
                normals[m] = _normals_lut[fid]

            return pts, normals

        raise ValueError(f"Unknown boundary_mode: {mode!r}")

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def sample(self) -> CollocationBatch3D:
        """Draw one full collocation batch.

        Returns
        -------
        CollocationBatch3D
            ``X_interior`` — (N_int, 3) interior PDE points.
            ``X_boundary`` — (N_bnd, 3) surface BC points.
            ``normals_boundary`` — (N_bnd, 3) outward normals (when available).
            ``boundary_groups`` — named groups when ``boundary_mode="bbox_planes"``.
        """
        cfg = self.cfg
        X_int = self.sample_interior(cfg.n_interior) if cfg.n_interior > 0 else np.zeros((0, 3))
        X_bnd, normals = self.sample_boundary(cfg.n_boundary) if cfg.n_boundary > 0 else (np.zeros((0, 3)), None)

        bnd_groups: Optional[Dict[str, np.ndarray]] = None
        if cfg.boundary_mode == "bbox_planes" and cfg.n_boundary > 0:
            plane_names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
            b0, b1 = self.mesh.bounds()
            tol = 1e-8
            fixed_vals = [b0[0], b1[0], b0[1], b1[1], b0[2], b1[2]]
            fixed_dims = [0, 0, 1, 1, 2, 2]
            bnd_groups = {}
            for name, dim, val in zip(plane_names, fixed_dims, fixed_vals):
                idx = np.where(np.abs(X_bnd[:, dim] - val) < tol)[0]
                if idx.size > 0:
                    bnd_groups[name] = idx

        return CollocationBatch3D(
            X_interior=X_int,
            X_boundary=X_bnd,
            normals_boundary=normals,
            boundary_groups=bnd_groups,
        )

    # ------------------------------------------------------------------
    # Convenience: to torch tensors
    # ------------------------------------------------------------------

    def sample_tensors(
        self,
        device: str = "cpu",
        dtype=None,
    ) -> Dict[str, "torch.Tensor"]:
        """Like :meth:`sample` but returns a dict of torch Tensors.

        Keys: ``"X_int"``, ``"X_bnd"``, ``"normals_bnd"`` (if available).
        """
        import torch  # local import to keep numpy-only path possible
        dt = dtype or torch.float32
        batch = self.sample()
        out = {
            "X_int": torch.tensor(batch.X_interior, dtype=dt, device=device),
            "X_bnd": torch.tensor(batch.X_boundary, dtype=dt, device=device),
        }
        if batch.normals_boundary is not None:
            out["normals_bnd"] = torch.tensor(batch.normals_boundary, dtype=dt, device=device)
        return out


__all__ = ["MeshCollocatorConfig", "CollocationBatch3D", "MeshCollocator"]
