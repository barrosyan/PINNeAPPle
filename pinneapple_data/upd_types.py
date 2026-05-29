"""Pure data-structure types for the UPD (Unified Physical Data) pipeline.

These classes are dependency-free within PINNeAPPle and can be safely
imported by any module.  UPDDataset (which requires PINNMapping from
pinneapple_pinn) lives in pinneapple_pinn.io.upd_dataset and imports from here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def seconds_since(t: np.ndarray, t0: Optional[np.datetime64] = None) -> np.ndarray:
    """Convert a ``datetime64`` array to float seconds elapsed since *t0*.

    *t0* defaults to ``t.min()`` when ``None``.  The result is a ``float64``
    array of the same shape as *t*.

    This utility belongs in ``pinneapple_data`` because it is a general-purpose
    time-coordinate helper used by any UPD consumer, not just PINN models.
    """
    t = np.asarray(t)
    if t0 is None:
        t0 = t.min()
    dt_ns = (
        t.astype("datetime64[ns]") - t0.astype("datetime64[ns]")
    ).astype("timedelta64[ns]").astype(np.int64)
    return dt_ns.astype(np.float64) / 1e9

Tensor = torch.Tensor


@dataclass
class UPDItem:
    """One UPD shard item: points to a Zarr store and a JSON metadata file."""

    zarr_path: str
    meta_path: str

    def load_meta(self) -> Dict[str, Any]:
        """Load JSON metadata from meta_path."""
        return json.loads(Path(self.meta_path).read_text(encoding="utf-8"))

    def open_dataset(self):
        """Open Zarr dataset at zarr_path (returns xr.Dataset)."""
        import xarray as xr
        return xr.open_zarr(self.zarr_path)


@dataclass
class ConditionSpec:
    """Sampling definition for a single PINN condition.

    type:
      - ``"initial"``  — t fixed to the first time step in the shard
      - ``"boundary"`` — lat/lon boundary edges
      - ``"interior"`` — random interior points
      - ``"slice"``    — fixed value on one coordinate: ``{"coord":"lev","value":850}``

    equation:
      Symbolic equation string; stored here for coordination/debugging only.
    """

    name: str
    type: str
    equation: str
    n: int = 1024
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingSpec:
    """Overall sampling plan for a UPD shard."""

    n_collocation: int = 4096
    conditions: List[ConditionSpec] = field(default_factory=list)
    n_data: int = 0          # supervised data points (0 = skip)
    replace: bool = True     # sampling with replacement
    seed: int = 0


@dataclass
class Batch:
    """PINN-ready training batch.

    Compatible with the loss_fn produced by ``pinneapple_pinn.compiler``:
      - collocation: tuple of input tensors for the PDE residual
      - conditions:  list of input-tensor tuples (one per ConditionSpec)
      - data:        (inputs, y_true) pair for supervised loss, or None
    """

    collocation: Optional[Tuple[Tensor, ...]] = None
    conditions: Optional[List[Tuple[Tensor, ...]]] = None
    data: Optional[Tuple[Tuple[Tensor, ...], Tensor]] = None


__all__ = ["seconds_since", "UPDItem", "ConditionSpec", "SamplingSpec", "Batch"]
