from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List, Literal, Callable

import numpy as np
import torch

from pinneaple_environment.spec import ProblemSpec


InsideMode = Literal["trimesh_contains", "voxel_occupancy", "bbox", "sdf"]
SDFMode = Literal["mesh_proximity", "callable"]


@dataclass(frozen=True)
class VoxelInsideConfig:
    pitch: Optional[float] = None
    target_resolution: int = 96
    fill: bool = True


@dataclass(frozen=True)
class SDFConfig:
    mode: SDFMode = "mesh_proximity"
    sdf_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None
    inside_if_leq: float = 0.0


@dataclass(frozen=True)
class TagHeuristics:
    enabled: bool = True
    inlet_outlet_axis: int = 0
    inlet_is_min: bool = True
    outlet_is_min: bool = False
    plane_eps_frac: float = 0.02
    require_normal_alignment: bool = False
    normal_alignment_cos: float = 0.6

    fixed_axis: Optional[int] = None
    fixed_is_min: bool = True

    enable_clustering: bool = False
    n_clusters: int = 4
    cluster_tag_map: Dict[int, str] = field(default_factory=dict)


@dataclass
class STLDomainBatchConfig:
    n_col: int = 60_000
    n_bc: int = 24_000
    n_ic: int = 0
    n_data: int = 0

    normalize_unit_box: bool = True

    inside_mode: InsideMode = "trimesh_contains"
    max_rejection_rounds: int = 60
    oversample_factor: int = 6

    voxel: VoxelInsideConfig = field(default_factory=VoxelInsideConfig)
    sdf: SDFConfig = field(default_factory=SDFConfig)

    surface_even: bool = True

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    tags: TagHeuristics = field(default_factory=TagHeuristics)


class STLDomainBatchBuilder:
    def __init__(self, cfg: STLDomainBatchConfig):
        self.cfg = cfg

    def _load_trimesh(self, stl_path: Union[str, Path]):
        import trimesh

        p = Path(stl_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)

        obj = trimesh.load(str(p), force="mesh")
        if isinstance(obj, trimesh.Scene):
            if not obj.geometry:
                raise ValueError(f"STL scene has no geometry: {p}")
            obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
        if not isinstance(obj, trimesh.Trimesh):
            raise TypeError(f"Loaded object is not trimesh.Trimesh: {type(obj).__name__}")

        try:
            obj.remove_duplicate_faces()
        except Exception:
            pass
        try:
            obj.remove_degenerate_faces()
        except Exception:
            pass
        try:
            obj.remove_unreferenced_vertices()
        except Exception:
            pass

        return obj

    def _normalize_trimesh(self, mesh):
        m = mesh.copy()
        bb = m.bounding_box
        center = bb.centroid
        ext = bb.extents
        scale = float(np.max(ext)) if ext is not None else 1.0
        m.apply_translation(-center)
        if self.cfg.normalize_unit_box and scale > 0:
            m.apply_scale(1.0 / scale)
        return m

    def _bbox(self, mesh) -> Tuple[np.ndarray, np.ndarray]:
        b = np.asarray(mesh.bounds, dtype=np.float32)
        return b[0], b[1]

    def _sample_surface_with_normals(self, mesh, n: int) -> Tuple[np.ndarray, np.ndarray]:
        import trimesh

        Xb, face_idx = trimesh.sample.sample_surface(mesh, n)
        Xb = np.asarray(Xb, dtype=np.float32)
        fn = np.asarray(mesh.face_normals, dtype=np.float32)
        Nb = fn[np.asarray(face_idx, dtype=np.int64)]
        Nb = Nb / (np.linalg.norm(Nb, axis=1, keepdims=True) + 1e-12)
        return Xb.astype(np.float32), Nb.astype(np.float32)

    def _inside_bbox(self, X: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
        return np.all((X >= bmin[None, :]) & (X <= bmax[None, :]), axis=1)

    def _inside_trimesh_contains(self, mesh, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        try:
            keep = mesh.contains(X.astype(np.float64))
            return np.asarray(keep, dtype=bool)
        except Exception as e:
            ctx.setdefault("warnings", []).append(f"mesh.contains failed: {e}")
            return np.zeros((X.shape[0],), dtype=bool)

    def _voxel_pitch(self, bmin: np.ndarray, bmax: np.ndarray) -> float:
        ext = bmax - bmin
        m = float(np.max(ext))
        res = max(8, int(self.cfg.voxel.target_resolution))
        pitch = float(m / res) if m > 0 else 0.05
        if self.cfg.voxel.pitch is not None:
            pitch = float(self.cfg.voxel.pitch)
        return max(pitch, 1e-6)

    def _inside_voxel_occupancy(self, mesh, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        import trimesh  # noqa: F401

        bmin, bmax = self._bbox(mesh)
        pitch = self._voxel_pitch(bmin, bmax)
        ctx.setdefault("voxel", {})["pitch"] = pitch

        try:
            vg = mesh.voxelized(pitch=pitch)
            if bool(self.cfg.voxel.fill):
                vg = vg.fill()
            keep = vg.is_filled(X.astype(np.float64))
            return np.asarray(keep, dtype=bool)
        except Exception as e:
            ctx.setdefault("warnings", []).append(f"voxel_occupancy failed: {e}")
            return np.zeros((X.shape[0],), dtype=bool)

    def _sdf_values(self, mesh, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if self.cfg.sdf.mode == "callable":
            if self.cfg.sdf.sdf_fn is None:
                raise ValueError("SDF mode 'callable' requires cfg.sdf.sdf_fn.")
            v = self.cfg.sdf.sdf_fn(X, ctx)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return v

        import trimesh
        try:
            v = trimesh.proximity.signed_distance(mesh, X.astype(np.float64))
            return np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception as e:
            ctx.setdefault("warnings", []).append(f"signed_distance failed: {e}")
            try:
                _, dist, _ = mesh.nearest.on_surface(X.astype(np.float64))
                dist = np.asarray(dist, dtype=np.float32).reshape(-1)
                return dist
            except Exception as e2:
                ctx.setdefault("warnings", []).append(f"nearest.on_surface failed: {e2}")
                return np.full((X.shape[0],), np.inf, dtype=np.float32)

    def _inside_sdf(self, mesh, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        sdf = self._sdf_values(mesh, X, ctx)
        thr = float(self.cfg.sdf.inside_if_leq)
        return (sdf <= thr)

    def _inside(self, mesh, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        mode = self.cfg.inside_mode
        if mode == "bbox":
            bmin, bmax = self._bbox(mesh)
            return self._inside_bbox(X, bmin, bmax)
        if mode == "trimesh_contains":
            if not bool(getattr(mesh, "is_watertight", False)):
                ctx.setdefault("warnings", []).append("inside_mode=trimesh_contains but mesh not watertight; consider voxel_occupancy.")
            return self._inside_trimesh_contains(mesh, X, ctx)
        if mode == "voxel_occupancy":
            return self._inside_voxel_occupancy(mesh, X, ctx)
        if mode == "sdf":
            return self._inside_sdf(mesh, X, ctx)
        raise ValueError(f"Unknown inside_mode: {mode}")

    def _sample_interior(self, mesh, n: int, bmin: np.ndarray, bmax: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(7)
        need = n
        chunks = []
        rounds = 0

        while need > 0 and rounds < self.cfg.max_rejection_rounds:
            rounds += 1
            m = max(10_000, need * self.cfg.oversample_factor)
            cand = rng.random((m, 3), dtype=np.float32) * (bmax - bmin) + bmin
            keep = self._inside(mesh, cand, ctx)
            pts = cand[keep]
            if pts.shape[0] == 0:
                continue
            take = pts[:need]
            chunks.append(take)
            need -= take.shape[0]

        if need > 0:
            ctx.setdefault("warnings", []).append(f"Interior sampling incomplete (missing {need}). Filling with bbox points.")
            fill = rng.random((need, 3), dtype=np.float32) * (bmax - bmin) + bmin
            chunks.append(fill)

        X = np.concatenate(chunks, axis=0).astype(np.float32)
        return X

    def _plane_mask(
        self,
        Xb: np.ndarray,
        Nb: np.ndarray,
        *,
        axis: int,
        is_min: bool,
        bmin: np.ndarray,
        bmax: np.ndarray,
        eps_frac: float,
        require_normal_alignment: bool,
        normal_alignment_cos: float,
    ) -> np.ndarray:
        extent = float(bmax[axis] - bmin[axis])
        eps = eps_frac * max(extent, 1e-12)
        plane_val = float(bmin[axis] if is_min else bmax[axis])
        dist = np.abs(Xb[:, axis] - plane_val)
        m = dist <= eps

        if require_normal_alignment:
            n_ref = np.zeros((3,), dtype=np.float32)
            n_ref[axis] = -1.0 if is_min else 1.0
            cos = np.sum(Nb * n_ref[None, :], axis=1)
            m = m & (cos >= normal_alignment_cos)

        return m

    def _cluster_tags(self, Xb: np.ndarray, n_clusters: int, ctx: Dict[str, Any]) -> np.ndarray:
        rng = np.random.default_rng(7)
        N = Xb.shape[0]
        k = max(2, int(n_clusters))
        centers = Xb[rng.integers(0, N, size=(k,))].copy()

        for _ in range(12):
            d2 = np.sum((Xb[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                m = labels == j
                if np.any(m):
                    new_centers[j] = np.mean(Xb[m], axis=0)
            if np.max(np.linalg.norm(new_centers - centers, axis=1)) < 1e-5:
                centers = new_centers
                break
            centers = new_centers

        ctx.setdefault("clusters", {})["centers"] = centers.astype(np.float32)
        return labels.astype(np.int64)

    def _make_tag_masks(self, Xb: np.ndarray, Nb: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, ctx: Dict[str, Any]) -> Dict[str, np.ndarray]:
        tags: Dict[str, np.ndarray] = {}
        tags["boundary"] = np.ones((Xb.shape[0],), dtype=bool)

        if not self.cfg.tags.enabled:
            return tags

        axis = int(self.cfg.tags.inlet_outlet_axis)
        inlet = self._plane_mask(
            Xb, Nb,
            axis=axis, is_min=bool(self.cfg.tags.inlet_is_min),
            bmin=bmin, bmax=bmax,
            eps_frac=float(self.cfg.tags.plane_eps_frac),
            require_normal_alignment=bool(self.cfg.tags.require_normal_alignment),
            normal_alignment_cos=float(self.cfg.tags.normal_alignment_cos),
        )
        outlet = self._plane_mask(
            Xb, Nb,
            axis=axis, is_min=bool(self.cfg.tags.outlet_is_min),
            bmin=bmin, bmax=bmax,
            eps_frac=float(self.cfg.tags.plane_eps_frac),
            require_normal_alignment=bool(self.cfg.tags.require_normal_alignment),
            normal_alignment_cos=float(self.cfg.tags.normal_alignment_cos),
        )

        tags["inlet"] = inlet
        tags["outlet"] = outlet
        tags["walls"] = tags["boundary"] & (~inlet) & (~outlet)

        if self.cfg.tags.fixed_axis is not None:
            fixed = self._plane_mask(
                Xb, Nb,
                axis=int(self.cfg.tags.fixed_axis),
                is_min=bool(self.cfg.tags.fixed_is_min),
                bmin=bmin, bmax=bmax,
                eps_frac=float(self.cfg.tags.plane_eps_frac),
                require_normal_alignment=bool(self.cfg.tags.require_normal_alignment),
                normal_alignment_cos=float(self.cfg.tags.normal_alignment_cos),
            )
            tags["fixed"] = fixed

        if self.cfg.tags.enable_clustering:
            labels = self._cluster_tags(Xb, int(self.cfg.tags.n_clusters), ctx)
            for j in range(int(self.cfg.tags.n_clusters)):
                tags[f"cluster_{j}"] = (labels == j)
            for j, name in self.cfg.tags.cluster_tag_map.items():
                j = int(j)
                if f"cluster_{j}" in tags:
                    tags[name] = tags.get(name, np.zeros_like(tags["boundary"])) | tags[f"cluster_{j}"]

        return tags

    def _targets_from_conditions(
        self,
        spec: ProblemSpec,
        Xb: np.ndarray,
        ctx: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        out_dim = len(spec.fields)
        y_bc = np.full((Xb.shape[0], out_dim), np.nan, dtype=np.float32)
        masks: Dict[str, np.ndarray] = {}

        ctx_local = dict(ctx)
        ctx_local.setdefault("tag_masks", {})

        for cond in spec.conditions:
            if cond.kind not in ("dirichlet", "neumann", "robin"):
                continue

            try:
                m = cond.mask(Xb, ctx_local)
                m = np.asarray(m, dtype=bool)
            except Exception as e:
                m = np.zeros((Xb.shape[0],), dtype=bool)
                ctx_local.setdefault("warnings", []).append(f"Condition mask failed for {cond.name}: {e}")

            masks[f"mask_{cond.name}"] = m

            if cond.kind == "dirichlet":
                if np.any(m):
                    X_sel = Xb[m]
                    vals = cond.values(X_sel, ctx_local)
                    vals = np.asarray(vals, dtype=np.float32)
                    if vals.ndim == 1:
                        vals = vals[:, None]

                    for j, fname in enumerate(cond.fields):
                        if fname not in spec.fields:
                            ctx_local.setdefault("warnings", []).append(f"Condition '{cond.name}' refers to unknown field '{fname}'.")
                            continue
                        idx = list(spec.fields).index(fname)
                        y_bc[m, idx] = vals[:, j]

        ctx.update(ctx_local)
        return y_bc, masks

    def build(
        self,
        spec: ProblemSpec,
        stl_path: Union[str, Path],
        *,
        user_ctx: Optional[Dict[str, Any]] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        x_ic: Optional[np.ndarray] = None,
        y_ic: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        if user_ctx:
            ctx.update(user_ctx)
        ctx.setdefault("warnings", [])

        mesh = self._load_trimesh(stl_path)
        mesh = self._normalize_trimesh(mesh)
        bmin, bmax = self._bbox(mesh)

        ctx["mesh_info"] = {
            "is_watertight": bool(getattr(mesh, "is_watertight", False)),
            "n_verts": int(mesh.vertices.shape[0]),
            "n_faces": int(mesh.faces.shape[0]),
        }
        ctx["bounds"] = {"min": bmin, "max": bmax}

        Xb, Nb = self._sample_surface_with_normals(mesh, int(self.cfg.n_bc))
        tag_masks = self._make_tag_masks(Xb, Nb, bmin, bmax, ctx)
        ctx["tag_masks"] = tag_masks

        Xc = self._sample_interior(mesh, int(self.cfg.n_col), bmin, bmax, ctx)
        y_bc, masks = self._targets_from_conditions(spec, Xb, ctx)

        if self.cfg.n_data > 0 and x_data is not None and y_data is not None:
            rng = np.random.default_rng(7)
            n = int(self.cfg.n_data)
            idx = rng.integers(0, x_data.shape[0], size=(n,))
            Xd = x_data[idx].astype(np.float32)
            Yd = y_data[idx].astype(np.float32)
        else:
            Xd = np.zeros((0, 3), dtype=np.float32)
            Yd = np.zeros((0, len(spec.fields)), dtype=np.float32)

        if x_ic is not None and y_ic is not None:
            Xi = x_ic.astype(np.float32)
            Yi = y_ic.astype(np.float32)
        else:
            Xi = np.zeros((0, Xc.shape[1]), dtype=np.float32)
            Yi = np.zeros((0, len(spec.fields)), dtype=np.float32)

        dev = torch.device(self.cfg.device)
        dtype = self.cfg.dtype

        batch: Dict[str, Any] = {
            "x_col": torch.as_tensor(Xc, device=dev, dtype=dtype),
            "x_bc": torch.as_tensor(Xb, device=dev, dtype=dtype),
            "y_bc": torch.as_tensor(y_bc, device=dev, dtype=dtype),
            "n_bc": torch.as_tensor(Nb, device=dev, dtype=dtype),
            "x_data": torch.as_tensor(Xd, device=dev, dtype=dtype),
            "y_data": torch.as_tensor(Yd, device=dev, dtype=dtype),
            "x_ic": torch.as_tensor(Xi, device=dev, dtype=dtype),
            "y_ic": torch.as_tensor(Yi, device=dev, dtype=dtype),
            "ctx": ctx,
        }

        for k, m in masks.items():
            batch[k] = torch.as_tensor(m, device=dev, dtype=torch.bool)

        return batch