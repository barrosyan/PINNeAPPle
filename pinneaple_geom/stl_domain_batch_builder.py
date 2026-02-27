from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np

try:
    import trimesh  # type: ignore
except Exception as e:  # pragma: no cover
    trimesh = None


@dataclass(frozen=True)
class STLDomainConfig:
    n_col: int = 50_000
    n_bc: int = 10_000
    n_data: int = 0
    seed: int = 0

    # Tagging / boundary classification
    inlet_outlet_axis: int = -1  # -1=auto, 0=x,1=y,2=z
    inlet_frac: float = 0.02
    outlet_frac: float = 0.02
    walls_tag: str = "walls"
    inlet_tag: str = "inlet"
    outlet_tag: str = "outlet"
    boundary_tag: str = "boundary"

    # sampling
    oversample_factor: float = 4.0
    boundary_oversample_factor: float = 2.0


def _auto_axis(bounds_min: np.ndarray, bounds_max: np.ndarray) -> int:
    ext = bounds_max - bounds_min
    return int(np.argmax(ext))


def _normalize_mesh(mesh):
    mesh = mesh.copy()
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = float(np.max(mesh.bounding_box.extents))
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh


def load_and_normalize_stl(stl_path: str):
    if trimesh is None:
        raise ImportError("trimesh is required for STL support. Install with: pip install 'pinneaple[geom]'")
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return _normalize_mesh(mesh)


def _sample_interior(mesh, n: int, rng: np.random.Generator) -> np.ndarray:
    bmin, bmax = mesh.bounds
    need = n
    pts = []
    tries = 0
    while need > 0 and tries < 60:
        m = int(max(need * 2, 20_000))
        cand = rng.random((m, 3)).astype(np.float32) * (bmax - bmin) + bmin
        inside = mesh.contains(cand)
        keep = cand[inside]
        if keep.shape[0] > 0:
            pts.append(keep[:need])
            need -= keep[:need].shape[0]
        tries += 1
    if need > 0:
        raise RuntimeError("Could not sample enough interior points. Check if STL is watertight/closed.")
    return np.concatenate(pts, axis=0).astype(np.float32)


def _sample_surface(mesh, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # returns (points, normals)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    pts = pts.astype(np.float32)
    normals = mesh.face_normals[face_idx].astype(np.float32)
    return pts, normals


def _tag_planes(
    Xb: np.ndarray,
    normals: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    axis: int,
    inlet_frac: float,
    outlet_frac: float,
    inlet_tag: str,
    outlet_tag: str,
    walls_tag: str,
    boundary_tag: str,
) -> Dict[str, np.ndarray]:
    # Basic geometric tagging:
    # - inlet/outlet based on coordinate percentile near min/max on chosen axis
    # - walls is the remainder
    # Also provide 'boundary' tag = all boundary points.
    coord = Xb[:, axis]
    lo = float(bounds_min[axis] + inlet_frac * (bounds_max[axis] - bounds_min[axis]))
    hi = float(bounds_max[axis] - outlet_frac * (bounds_max[axis] - bounds_min[axis]))

    inlet = coord <= lo
    outlet = coord >= hi
    walls = ~(inlet | outlet)

    return {
        boundary_tag: np.ones((Xb.shape[0],), dtype=bool),
        inlet_tag: inlet,
        outlet_tag: outlet,
        walls_tag: walls,
    }


def build_stl_domain_batch(
    stl_path: str,
    cfg: STLDomainConfig,
    *,
    include_time: bool = False,
    t_span: Tuple[float, float] = (0.0, 1.0),
    ctx_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a single dict-batch suitable for the Trainer/Compiler.

    Produces:
      - x_col: collocation points in the interior
      - x_bc: boundary points on surface
      - n_bc: surface normals
      - ctx: bounds + tag masks
    """
    rng = np.random.default_rng(cfg.seed)

    mesh = load_and_normalize_stl(stl_path)
    bmin, bmax = mesh.bounds
    axis = cfg.inlet_outlet_axis if cfg.inlet_outlet_axis in (0, 1, 2) else _auto_axis(bmin, bmax)

    Xc = _sample_interior(mesh, cfg.n_col, rng)
    Xb, Nb = _sample_surface(mesh, int(cfg.n_bc * cfg.boundary_oversample_factor), rng)
    # downsample to n_bc while keeping inlet/outlet coverage
    tags_tmp = _tag_planes(Xb, Nb, bmin, bmax, axis, cfg.inlet_frac, cfg.outlet_frac, cfg.inlet_tag, cfg.outlet_tag, cfg.walls_tag, cfg.boundary_tag)

    # ensure some inlet/outlet points exist
    inlet_idx = np.where(tags_tmp[cfg.inlet_tag])[0]
    outlet_idx = np.where(tags_tmp[cfg.outlet_tag])[0]
    walls_idx = np.where(tags_tmp[cfg.walls_tag])[0]

    # allocate roughly
    n_in = max(1, int(0.15 * cfg.n_bc)) if inlet_idx.size > 0 else 0
    n_out = max(1, int(0.15 * cfg.n_bc)) if outlet_idx.size > 0 else 0
    n_w = cfg.n_bc - n_in - n_out

    sel = []
    if n_in > 0:
        sel.append(rng.choice(inlet_idx, size=min(n_in, inlet_idx.size), replace=inlet_idx.size < n_in))
    if n_out > 0:
        sel.append(rng.choice(outlet_idx, size=min(n_out, outlet_idx.size), replace=outlet_idx.size < n_out))
    if n_w > 0:
        sel.append(rng.choice(walls_idx, size=min(n_w, walls_idx.size), replace=walls_idx.size < n_w))
    sel = np.concatenate(sel) if sel else np.arange(min(cfg.n_bc, Xb.shape[0]))
    rng.shuffle(sel)

    Xb = Xb[sel].astype(np.float32)
    Nb = Nb[sel].astype(np.float32)
    tags = _tag_planes(Xb, Nb, bmin, bmax, axis, cfg.inlet_frac, cfg.outlet_frac, cfg.inlet_tag, cfg.outlet_tag, cfg.walls_tag, cfg.boundary_tag)

    # add time coordinate if required
    if include_time:
        t0, t1 = t_span
        tc = rng.random((Xc.shape[0], 1)).astype(np.float32) * (t1 - t0) + t0
        tb = rng.random((Xb.shape[0], 1)).astype(np.float32) * (t1 - t0) + t0
        Xc = np.concatenate([Xc, tc], axis=1)
        Xb = np.concatenate([Xb, tb], axis=1)

    ctx = {
        "bounds": {"min": bmin.astype(np.float32).tolist(), "max": bmax.astype(np.float32).tolist()},
        "tag_masks": tags,
        "stl_path": stl_path,
        "axis": axis,
    }
    if ctx_extra:
        ctx.update(ctx_extra)

    batch = {
        "x_col": torch_from_numpy(Xc),
        "x_bc": torch_from_numpy(Xb),
        "n_bc": torch_from_numpy(Nb),
        "ctx": ctx,
    }
    return batch


def torch_from_numpy(x: np.ndarray):
    import torch
    return torch.from_numpy(np.asarray(x, dtype=np.float32))