"""PINN batch builders: convert various data sources into the unified dict-batch format.

Expected output format (consumed by pinneaple_pinn.compiler):
  x_col, x_bc, y_bc, x_ic, y_ic, x_data, y_data, ctx
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch


def _to_tensor(
    x: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if x is None:
        raise ValueError("Cannot convert None to tensor")
    t = x if torch.is_tensor(x) else (torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x))
    t = t.to(dtype=dtype)
    return t.to(device) if device is not None else t


def _sample_df(df: Any, n: int, cols: Tuple[str, ...]) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros((0, len(cols)), dtype=np.float32)
    n_eff = min(int(n), len(df))
    samp = df.sample(n=n_eff, replace=(len(df) < n_eff))
    return samp.loc[:, list(cols)].to_numpy(dtype=np.float32)


@dataclass
class PINNBatch:
    """In-memory PINN-ready batch (thin wrapper around the dict)."""

    batch: Dict[str, Any]


def build_from_bundle(
    bundle: Any,
    *,
    n_collocation: int = 4096,
    n_boundary: int = 2048,
    n_data: int = 2048,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PINNBatch:
    """Create a PINN-style batch from an Arena BundleData.

    Parameters
    ----------
    bundle:
        ``pinneaple_arena.bundle.loader.BundleData`` instance (typed as Any to
        avoid a hard dependency on pinneaple_arena).
    """
    x_col = _to_tensor(_sample_df(bundle.points_collocation, n_collocation, ("x", "y")), device=device, dtype=dtype)
    x_bc_np = _sample_df(bundle.points_boundary, n_boundary, ("x", "y"))
    x_bc = _to_tensor(x_bc_np, device=device, dtype=dtype)
    y_bc = torch.zeros((x_bc.shape[0], 1), device=x_bc.device, dtype=x_bc.dtype)

    x_data = torch.zeros((0, 2), device=x_col.device, dtype=x_col.dtype)
    y_data = torch.zeros((0, 1), device=x_col.device, dtype=x_col.dtype)
    if bundle.sensors is not None and len(bundle.sensors) > 0 and n_data > 0:
        s = bundle.sensors
        cols_xy = [c for c in ("x", "y") if c in s.columns]
        if len(cols_xy) == 2:
            s_samp = s.sample(n=min(n_data, len(s)), replace=(len(s) < n_data))
            x_data = _to_tensor(s_samp[cols_xy].to_numpy(dtype=np.float32), device=device, dtype=dtype)
            out_cols = [c for c in ("u", "v", "p", "T") if c in s_samp.columns]
            if out_cols:
                y_data = _to_tensor(s_samp[out_cols].to_numpy(dtype=np.float32), device=device, dtype=dtype)

    return PINNBatch(batch={
        "x_col": x_col, "x_bc": x_bc, "y_bc": y_bc,
        "x_data": x_data, "y_data": y_data,
        "ctx": {"manifest": dict(bundle.manifest), "conditions": dict(bundle.conditions)},
    })


def build_from_solver(
    problem_spec: Any,
    geometry: Any,
    solver_cfg: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PINNBatch:
    """Build a PINN batch by running a solver or using pre-computed tensors.

    solver_cfg may contain:
      - ``solver_fn``: callable(problem_spec, geometry, cfg) → dict
      - or the batch keys directly (``x_col``, ``x_bc``, …)
    """
    if "solver_fn" in solver_cfg and callable(solver_cfg["solver_fn"]):
        data = solver_cfg["solver_fn"](problem_spec=problem_spec, geometry=geometry, cfg=solver_cfg)
    else:
        data = {k: solver_cfg.get(k) for k in
                ("x_col", "x_bc", "y_bc", "x_ic", "y_ic", "x_data", "y_data", "ctx")
                if k in solver_cfg}

    if "x_col" not in data or data["x_col"] is None:
        raise ValueError("build_from_solver requires 'x_col' (collocation points).")

    batch: Dict[str, Any] = {}
    for k in ("x_col", "x_bc", "y_bc", "x_ic", "y_ic", "x_data", "y_data"):
        if k in data and data[k] is not None:
            batch[k] = _to_tensor(data[k], device=device, dtype=dtype)
    batch.setdefault("ctx", data.get("ctx", {}))
    return PINNBatch(batch=batch)


def build_from_real_data(
    adapter_cfg: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> PINNBatch:
    """Build a PINN batch from real measurement data.

    adapter_cfg may contain:
      - ``loader_fn``: callable(cfg) → dict with ``x_data`` / ``y_data``
      - or ``x_data`` / ``y_data`` tensors directly
    """
    if "loader_fn" in adapter_cfg and callable(adapter_cfg["loader_fn"]):
        data = adapter_cfg["loader_fn"](cfg=adapter_cfg)
    else:
        data = dict(adapter_cfg)

    if "x_data" not in data or "y_data" not in data:
        raise ValueError("build_from_real_data requires 'x_data' and 'y_data'.")

    batch: Dict[str, Any] = {
        "x_data": _to_tensor(data["x_data"], device=device, dtype=dtype),
        "y_data": _to_tensor(data["y_data"], device=device, dtype=dtype),
        "ctx": dict(data.get("ctx", {})),
    }
    for k in ("x_col", "x_bc", "y_bc", "x_ic", "y_ic"):
        if k in data and data[k] is not None:
            batch[k] = _to_tensor(data[k], device=device, dtype=dtype)

    return PINNBatch(batch=batch)


__all__ = ["PINNBatch", "build_from_bundle", "build_from_solver", "build_from_real_data"]
