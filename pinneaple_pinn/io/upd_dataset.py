"""UPD dataset loader and PINN-ready sampling (collocation, conditions, data)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

from .mappings import PINNMapping


Tensor = torch.Tensor


@dataclass
class UPDItem:
    """
    One UPD shard item: points to Zarr and JSON metadata.
    """
    zarr_path: str
    meta_path: str

    def load_meta(self) -> Dict[str, Any]:
        """Load JSON metadata from meta_path."""
        return json.loads(Path(self.meta_path).read_text(encoding="utf-8"))

    def open_dataset(self) -> xr.Dataset:
        """Open Zarr dataset at zarr_path."""
        return xr.open_zarr(self.zarr_path)


@dataclass
class ConditionSpec:
    """
    Sampling definition for a condition.

    type:
      - "initial": t fixed to first time in shard
      - "boundary": lat/lon boundaries (edges)
      - "interior": random interior points (often used as extra constraints)
      - "slice": any slice on a coord: {"coord":"lev","value":850}

    equation:
      symbolic equation string (used by PINNFactory separately)
      stored here only for coordination / debugging.
    """
    name: str
    type: str
    equation: str
    n: int = 1024
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingSpec:
    """
    Overall sampling plan for a UPD shard.
    """
    n_collocation: int = 4096
    conditions: List[ConditionSpec] = field(default_factory=list)
    # supervised data points (optional)
    n_data: int = 0
    # allow sampling with replacement
    replace: bool = True
    seed: int = 0


@dataclass
class Batch:
    """
    A training batch compatible with PINNFactory loss_fn:
      - collocation: tuple(inputs...) for PDE residual
      - conditions: list of tuple(inputs...) for each condition
      - data: (tuple(inputs...), y_true) if provided
    """
    collocation: Optional[Tuple[Tensor, ...]] = None
    conditions: Optional[List[Tuple[Tensor, ...]]] = None
    data: Optional[Tuple[Tuple[Tensor, ...], Tensor]] = None


class UPDDataset:
    """
    Reads UPD (Zarr + JSON) and produces PINN-ready samples:
      - collocation points (inputs only)
      - condition points (inputs only; one list entry per condition)
      - optional supervised data pairs from UPD variables

    Notes:
      - This MVP samples points directly from the gridded data.
      - For atmosphere reanalysis, independent vars typically come from coords:
          time, lat, lon, (optional lev)
      - Dependent vars (targets) come from UPD variables via mapping.
    """

    def __init__(
        self,
        item: UPDItem,
        mapping: PINNMapping,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.item = item
        self.mapping = mapping
        self.device = torch.device(device)
        self.dtype = dtype

        self.meta = item.load_meta()
        self.ds = item.open_dataset()

        # standard assumption: coords exist
        self._time0 = self.mapping.infer_time0(self.ds)

    # -----------------------
    # Public API
    # -----------------------
    def sample(self, spec: SamplingSpec) -> Batch:
        """Sample collocation, condition, and optional data points per spec."""
        rng = np.random.default_rng(spec.seed)

        collocation = None
        if spec.n_collocation > 0:
            collocation = self._sample_points(rng, n=spec.n_collocation, mode="interior")

        cond_batches = []
        for c in spec.conditions:
            cond_batches.append(self._sample_condition(rng, c))
        conditions = cond_batches if cond_batches else None

        data = None
        if spec.n_data and spec.n_data > 0:
            data = self._sample_data(rng, n=spec.n_data)

        return Batch(collocation=collocation, conditions=conditions, data=data)

    def get_problem_dims(self) -> Dict[str, Any]:
        """
        Convenience helper: returns independent/dependent vars for creating PINNProblemSpec.
        """
        return {
            "independent_vars": self.mapping.independent_vars(),
            "dependent_vars": self.mapping.dependent_vars(),
            "available_vars": list(self.ds.data_vars),
            "coords": list(self.ds.coords),
        }

    # -----------------------
    # Internals: sampling
    # -----------------------
    def _coord_arrays(self) -> Dict[str, np.ndarray]:
        """Build coordinate arrays (transformed/normalized) for independent vars."""
        return self.mapping.coord.make_coord_arrays(self.ds, time0=self._time0)

    def _random_index(self, rng, size: int, n: int, replace: bool) -> np.ndarray:
        """Sample n indices from [0, size); with or without replacement."""
        if size <= 0:
            raise ValueError("Cannot sample from empty dimension.")
        if replace:
            return rng.integers(0, size, size=n, endpoint=False)
        n = min(n, size)
        return rng.choice(size, size=n, replace=False)

    def _mesh_pick(
        self,
        rng,
        n: int,
        mode: str,
        replace: bool = True,
        slice_override: Optional[Dict[str, int]] = None,
        boundary_axis: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Picks random indices for each coord dimension needed by mapping.
        Returns dict {coord_name: idx_array}.
        """
        coords = self._coord_arrays()

        # Map independent vars to coordinate sources (e.g., t->time)
        coord_src = self.mapping.coord.coord_sources

        # Build index arrays for each coord used
        idx: Dict[str, np.ndarray] = {}
        for ind in self.mapping.independent_vars():
            src = coord_src[ind]
            arr = self.ds.coords[src].values
            dim_size = arr.shape[0]

            if slice_override and src in slice_override:
                # fixed index
                idx[src] = np.full((n,), int(slice_override[src]), dtype=np.int64)
                continue

            if mode == "initial" and src == "time":
                idx[src] = np.zeros((n,), dtype=np.int64)
                continue

            if mode == "boundary" and (src in ("lat", "lon")) and (boundary_axis == src):
                # choose either lower edge or upper edge randomly
                edge = rng.integers(0, 2, size=n)
                idx[src] = np.where(edge == 0, 0, dim_size - 1).astype(np.int64)
                continue

            idx[src] = self._random_index(rng, dim_size, n, replace=replace)

        return idx

    def _gather_inputs(self, idx: Dict[str, np.ndarray]) -> Tuple[Tensor, ...]:
        """
        Given sampled indices for coords, gather independent variables as tensors (N,1).
        """
        coords = self._coord_arrays()
        coord_src = self.mapping.coord.coord_sources

        tensors: List[Tensor] = []
        for ind in self.mapping.independent_vars():
            src = coord_src[ind]
            arr = coords[ind]  # already transformed & normalized for this independent var
            x = arr[idx[src]]
            t = torch.tensor(x, device=self.device, dtype=self.dtype).reshape(-1, 1)
            tensors.append(t)
        return tuple(tensors)

    def _sample_points(self, rng, n: int, mode: str, replace: bool = True) -> Tuple[Tensor, ...]:
        idx = self._mesh_pick(rng, n=n, mode=mode, replace=replace)
        return self._gather_inputs(idx)

    def _sample_condition(self, rng, cond: ConditionSpec) -> Tuple[Tensor, ...]:
        """Sample points for a single condition (initial, boundary, slice, interior)."""
        ctype = cond.type.lower().strip()
        n = int(cond.n)

        if ctype == "initial":
            idx = self._mesh_pick(rng, n=n, mode="initial", replace=True)
            return self._gather_inputs(idx)

        if ctype == "boundary":
            # MVP: choose either lat edge or lon edge, or user specifies axis
            axis = cond.options.get("axis")
            if axis not in (None, "lat", "lon"):
                raise ValueError("boundary axis must be 'lat' or 'lon' (or omitted).")
            if axis is None:
                axis = "lat" if rng.random() < 0.5 else "lon"
            idx = self._mesh_pick(rng, n=n, mode="boundary", replace=True, boundary_axis=axis)
            return self._gather_inputs(idx)

        if ctype == "slice":
            # Example: {"coord":"lev","value":850} -> pick nearest lev index
            coord = cond.options.get("coord")
            value = cond.options.get("value")
            if coord is None or value is None:
                raise ValueError("slice condition needs options: {coord, value}")
            if coord not in self.ds.coords:
                raise KeyError(f"Dataset missing coord '{coord}' for slice condition.")
            arr = self.ds.coords[coord].values.astype(float)
            j = int(np.argmin(np.abs(arr - float(value))))
            idx = self._mesh_pick(rng, n=n, mode="interior", replace=True, slice_override={coord: j})
            return self._gather_inputs(idx)

        # default: interior
        idx = self._mesh_pick(rng, n=n, mode="interior", replace=True)
        return self._gather_inputs(idx)

    def _sample_data(self, rng, n: int) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """
        Supervised data: sample points and fetch variable values as y_true.
        For MVP: y_true is concatenation of dependent vars (N, n_dep).
        """
        # pick indices for coords used by mapping
        idx = self._mesh_pick(rng, n=n, mode="interior", replace=True)

        # inputs
        x = self._gather_inputs(idx)

        # targets: gather from UPD variables (nearest grid point by index)
        ds_vars = self.mapping.var.pick_dataset_vars(self.ds)

        # Determine broadcast dims order from first var
        dep_vals = []
        for dep in self.mapping.dependent_vars():
            upd_name = self.mapping.var.upd_to_dep[dep]
            da = ds_vars[upd_name]

            # MVP assumption: da dims subset of (time, lev, lat, lon) or (time, lat, lon)
            # We'll index by coords present in da
            sel = {}
            # map from coord name to sampled indices
            # NOTE: coord names in dataset are standard "time/lat/lon/lev"
            for c in ["time", "lev", "lat", "lon"]:
                if c in da.dims:
                    if c not in idx:
                        # if mapping doesn't include lev but da has lev, pick random lev
                        # (this is a policy choice; user can set mapping/use_lev or slice conditions)
                        arr = self.ds.coords[c].values
                        sel[c] = rng.integers(0, arr.shape[0], size=n, endpoint=False)
                    else:
                        sel[c] = idx[c]

            # advanced indexing via isel
            y = da.isel(**{k: xr.DataArray(v, dims=("samples",)) for k, v in sel.items()}).values
            y = np.asarray(y).reshape(n, -1)  # ensure (N,1)
            dep_vals.append(y)

        y_true = np.concatenate(dep_vals, axis=1)  # (N, n_dep)
        y_true_t = torch.tensor(y_true, device=self.device, dtype=self.dtype)

        return x, y_true_t
