"""Arena dataset builder.

This module bridges:
  - Arena bundles on disk (BundleData)
  - Solver-generated datasets
  - Real measurement datasets

into the unified dict-batch format expected by ``pinneaple_pinn.compiler``:

  x_col, x_bc, y_bc, x_ic, y_ic, x_data, y_data, ctx

This keeps the Arena orchestration stable, while different tasks/sources can
plug in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from pinneaple_arena.bundle.loader import BundleData


def _to_tensor(x: Any, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if x is None:
        raise ValueError("Cannot convert None to tensor")
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _sample_df(df, n: int, cols: Tuple[str, ...]) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.zeros((0, len(cols)), dtype=np.float32)
    n_eff = min(int(n), len(df))
    samp = df.sample(n=n_eff, replace=(len(df) < n_eff))
    return samp.loc[:, list(cols)].to_numpy(dtype=np.float32)


@dataclass
class BundleDataLike:
    """In-memory representation of a normalized training dataset."""

    batch: Dict[str, Any]


def build_from_bundle(
    bundle: BundleData,
    *,
    n_collocation: int = 4096,
    n_boundary: int = 2048,
    n_data: int = 2048,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> BundleDataLike:
    """Create a PINN-style batch from an Arena BundleData.

    - x_col: sampled from points_collocation (x,y)
    - x_bc: sampled from points_boundary (x,y)
      y_bc: inferred from conditions.json when possible (otherwise omitted)
    - x_data/y_data: sampled from sensors.parquet if present
    - ctx: carries manifest/conditions

    Notes
    -----
    The Arena's default bundles are for Navier-Stokes steady 2D; sensors columns
    typically include u,v,p. This builder keeps the data flexible: it will
    populate y_data only if those columns exist.
    """

    # Collocation points
    x_col_np = _sample_df(bundle.points_collocation, n_collocation, ("x", "y"))
    x_col = _to_tensor(x_col_np, device=device, dtype=dtype)

    # Boundary points
    x_bc_np = _sample_df(bundle.points_boundary, n_boundary, ("x", "y"))
    x_bc = _to_tensor(x_bc_np, device=device, dtype=dtype)

    # Boundary targets: best-effort from conditions.json
    # (Many tasks will override / provide masks per region; for MVP we keep y_bc as zeros
    #  unless explicit values are present.)
    y_bc = torch.zeros((x_bc.shape[0], 1), device=x_bc.device, dtype=x_bc.dtype)

    # Data/sensors
    x_data = torch.zeros((0, 2), device=x_col.device, dtype=x_col.dtype)
    y_data = torch.zeros((0, 1), device=x_col.device, dtype=x_col.dtype)
    if bundle.sensors is not None and len(bundle.sensors) > 0 and n_data > 0:
        s = bundle.sensors
        cols_xy = [c for c in ("x", "y") if c in s.columns]
        if len(cols_xy) == 2:
            s_samp = s.sample(n=min(n_data, len(s)), replace=(len(s) < n_data))
            x_data = _to_tensor(s_samp[cols_xy].to_numpy(dtype=np.float32), device=device, dtype=dtype)

            # pick output columns heuristically
            out_cols = [c for c in ("u", "v", "p", "T") if c in s_samp.columns]
            if out_cols:
                y_data = _to_tensor(s_samp[out_cols].to_numpy(dtype=np.float32), device=device, dtype=dtype)

    batch: Dict[str, Any] = {
        "x_col": x_col,
        "x_bc": x_bc,
        "y_bc": y_bc,
        # optional
        "x_data": x_data,
        "y_data": y_data,
        "ctx": {
            "manifest": dict(bundle.manifest),
            "conditions": dict(bundle.conditions),
        },
    }
    return BundleDataLike(batch=batch)


def build_from_solver(
    problem_spec: Any,
    geometry: Any,
    solver_cfg: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> BundleDataLike:
    """Build dataset by running a solver.

    This is intentionally generic: the solver interface can vary.

    Expected solver_cfg keys (MVP):
      - 'solver_fn': callable returning a dict with any of:
          x_col, x_bc, y_bc, x_ic, y_ic, x_data, y_data, ctx
      - or provide those tensors directly in solver_cfg.
    """
    if "solver_fn" in solver_cfg and callable(solver_cfg["solver_fn"]):
        data = solver_cfg["solver_fn"](problem_spec=problem_spec, geometry=geometry, cfg=solver_cfg)
    else:
        # assume tensors provided directly
        data = {k: solver_cfg.get(k) for k in ("x_col", "x_bc", "y_bc", "x_ic", "y_ic", "x_data", "y_data", "ctx") if k in solver_cfg}

    if "x_col" not in data or data["x_col"] is None:
        raise ValueError("build_from_solver requires 'x_col' (collocation points) from solver_fn or solver_cfg.")

    batch: Dict[str, Any] = {}
    for k in ("x_col", "x_bc", "y_bc", "x_ic", "y_ic", "x_data", "y_data"):
        if k in data and data[k] is not None:
            batch[k] = _to_tensor(data[k], device=device, dtype=dtype)
    batch.setdefault("ctx", data.get("ctx", {}))
    return BundleDataLike(batch=batch)


def build_from_real_data(
    adapter_cfg: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> BundleDataLike:
    """Build dataset from real data.

    Expected adapter_cfg keys (MVP):
      - 'loader_fn': callable returning dict with keys like x_data/y_data (+ optionally x_col etc)
      - OR direct tensors in adapter_cfg.
    """
    if "loader_fn" in adapter_cfg and callable(adapter_cfg["loader_fn"]):
        data = adapter_cfg["loader_fn"](cfg=adapter_cfg)
    else:
        data = dict(adapter_cfg)

    if "x_data" not in data or "y_data" not in data:
        raise ValueError("build_from_real_data requires 'x_data' and 'y_data' (measurement inputs/targets).")

    batch: Dict[str, Any] = {
        "x_data": _to_tensor(data["x_data"], device=device, dtype=dtype),
        "y_data": _to_tensor(data["y_data"], device=device, dtype=dtype),
        "ctx": dict(data.get("ctx", {})),
    }

    # optional PINN components if user provided
    for k in ("x_col", "x_bc", "y_bc", "x_ic", "y_ic"):
        if k in data and data[k] is not None:
            batch[k] = _to_tensor(data[k], device=device, dtype=dtype)

    return BundleDataLike(batch=batch)
