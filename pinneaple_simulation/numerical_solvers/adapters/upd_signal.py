"""UPD signal/sample adapter for solver field extraction and stacking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def _as_tensor(x: Any, device=None, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if device is not None:
        t = t.to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def list_field_vars(sample: Any) -> List[str]:
    """
    Tries to list available physical field variables from a PhysicalSample-like object.

    Expected (duck-typed):
      - sample.fields: Dict[str, Tensor/ndarray]
        OR
      - sample.state: Dict[str, Tensor/ndarray]
    """
    if hasattr(sample, "fields") and isinstance(sample.fields, dict):
        return sorted(list(sample.fields.keys()))
    if hasattr(sample, "state") and isinstance(sample.state, dict):
        return sorted(list(sample.state.keys()))
    return []


def _get_field(sample: Any, var: str) -> Any:
    if hasattr(sample, "fields") and isinstance(sample.fields, dict) and var in sample.fields:
        return sample.fields[var]
    if hasattr(sample, "state") and isinstance(sample.state, dict) and var in sample.state:
        return sample.state[var]
    raise KeyError(f"Variable '{var}' not found. Available: {list_field_vars(sample)}")


def list_signal_axes(sample: Any, var: str) -> List[str]:
    """
    Best-effort axis naming for a field.
    If sample has sample.axes[var] or sample.dims[var], use it.
    Fallback: ["dim0","dim1",...]
    """
    if hasattr(sample, "axes") and isinstance(sample.axes, dict) and var in sample.axes:
        ax = sample.axes[var]
        if isinstance(ax, (list, tuple)) and all(isinstance(a, str) for a in ax):
            return list(ax)

    if hasattr(sample, "dims") and isinstance(sample.dims, dict) and var in sample.dims:
        ax = sample.dims[var]
        if isinstance(ax, (list, tuple)) and all(isinstance(a, str) for a in ax):
            return list(ax)

    field = _get_field(sample, var)
    t = _as_tensor(field)
    return [f"dim{i}" for i in range(t.ndim)]


def extract_1d_signal(
    sample: Any,
    *,
    var: str,
    axis: Union[int, str] = -1,
    reduce: str = "mean",
    index: Optional[int] = None,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract a 1D signal from a (possibly multi-dim) physical field.

    Options:
      - choose axis (int or axis name if available)
      - either:
          * reduce all other axes (mean/sum/median/max/min)
          * OR pick a fixed index for all other axes via `index` (single int for simplicity)

    Returns:
      signal: (T,)
      meta: includes chosen axis name, original shape, reduction mode, and dt if sample provides it
    """
    field = _get_field(sample, var)
    x = _as_tensor(field, device=device, dtype=dtype)

    axes = list_signal_axes(sample, var)

    if isinstance(axis, str):
        if axis not in axes:
            raise ValueError(f"axis='{axis}' not in axes {axes}")
        ax = axes.index(axis)
        axis_name = axis
    else:
        ax = int(axis)
        axis_name = axes[ax] if -x.ndim <= ax < x.ndim else f"dim{ax}"

    # move target axis to last
    x = x.movedim(ax, -1)

    # collapse other dims
    if x.ndim > 1:
        if index is not None:
            # pick same index on every other axis (MVP)
            for d in range(x.ndim - 1):
                idx = int(index)
                idx = max(0, min(idx, x.shape[d] - 1))
                x = x.select(dim=d, index=idx)
        else:
            red = (reduce or "mean").lower().strip()
            dims = tuple(range(x.ndim - 1))
            if red == "mean":
                x = x.mean(dim=dims)
            elif red == "sum":
                x = x.sum(dim=dims)
            elif red == "median":
                x = x.median(dim=dims[0]).values if len(dims) == 1 else x.flatten(0, -2).median(dim=0).values
            elif red == "max":
                x = x.amax(dim=dims)
            elif red == "min":
                x = x.amin(dim=dims)
            else:
                raise ValueError("reduce must be one of: mean,sum,median,max,min")

    signal = x.contiguous().view(-1)

    meta: Dict[str, Any] = {
        "var": var,
        "axis": axis_name,
        "reduce": reduce,
        "index": index,
        "original_shape": tuple(_as_tensor(field).shape),
        "signal_len": int(signal.shape[0]),
    }

    # Optional dt / time spacing
    # - sample.dt
    # - sample.meta.get("dt")
    if hasattr(sample, "dt"):
        meta["dt"] = float(sample.dt)
    elif hasattr(sample, "meta") and isinstance(sample.meta, dict) and "dt" in sample.meta:
        meta["dt"] = float(sample.meta["dt"])

    return signal, meta


def to_signal_batch(
    samples: List[Any],
    *,
    var: str,
    axis: Union[int, str] = -1,
    reduce: str = "mean",
    index: Optional[int] = None,
    device=None,
    dtype=None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Convert a list of samples into a padded batch tensor (B,Tmax).

    Returns:
      xb: (B,Tmax) padded with zeros
      metas: list of dicts (per sample)
    """
    sigs = []
    metas = []
    maxT = 0
    for s in samples:
        sig, meta = extract_1d_signal(s, var=var, axis=axis, reduce=reduce, index=index, device=device, dtype=dtype)
        sigs.append(sig)
        metas.append(meta)
        maxT = max(maxT, int(sig.shape[0]))

    xb = torch.zeros((len(sigs), maxT), device=device or sigs[0].device, dtype=dtype or sigs[0].dtype)
    for i, sig in enumerate(sigs):
        xb[i, : sig.shape[0]] = sig
        metas[i]["padded_to"] = maxT
    return xb, metas
