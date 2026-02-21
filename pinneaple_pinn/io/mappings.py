"""Mappings between UPD coordinates/variables and PINN independent/dependent vars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr


def seconds_since(t: np.ndarray, t0: Optional[np.datetime64] = None) -> np.ndarray:
    """
    Convert datetime64 array to float seconds since t0 (defaults to min(t)).
    """
    t = np.asarray(t)
    if t0 is None:
        t0 = t.min()
    # convert to ns int then seconds
    dt_ns = (t.astype("datetime64[ns]") - t0.astype("datetime64[ns]")).astype("timedelta64[ns]").astype(np.int64)
    return dt_ns.astype(np.float64) / 1e9


@dataclass
class CoordMapping:
    """
    Mapping from UPD coords -> PINN independent variables.

    Example:
      ind_vars = ["t","x","y"] mapping:
        - t <- UPD time (seconds since shard start)
        - x <- lon (degrees or radians)
        - y <- lat

    norm:
      - optional normalization: (x - mean)/std or min-max to [-1,1]
      - for MVP, we provide simple min-max scaling to [-1,1]
    """
    ind_vars: List[str]
    coord_sources: Dict[str, str]  # e.g. {"t":"time","lat":"lat","lon":"lon","lev":"lev"}
    coord_transform: Dict[str, str] = field(default_factory=dict)
    # coord_transform values: "identity" | "seconds_since_start" | "deg2rad" | "minmax_-1_1"
    normalize_to_unit: bool = True

    def _apply_transform(
        self, name: str, arr: np.ndarray, ref: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply coord_transform for name (identity, deg2rad, seconds_since_start) and optional normalize."""
        tfm = (self.coord_transform.get(name) or "identity").lower()
        if tfm == "identity":
            out = arr.astype(np.float64)
        elif tfm == "deg2rad":
            out = np.deg2rad(arr.astype(np.float64))
        elif tfm == "seconds_since_start":
            # expects datetime64 array
            out = seconds_since(arr, t0=ref.get("_time0"))
        else:
            out = arr.astype(np.float64)

        if self.normalize_to_unit:
            # min-max to [-1,1] (safe and common for PINNs)
            mn = np.nanmin(out)
            mx = np.nanmax(out)
            if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
                out = 2.0 * (out - mn) / (mx - mn) - 1.0
        return out

    def make_coord_arrays(
        self, ds: xr.Dataset, time0: Optional[np.datetime64] = None
    ) -> Dict[str, np.ndarray]:
        """Build coord arrays for each independent var from dataset (transformed)."""
        ref = {"_time0": time0}
        out: Dict[str, np.ndarray] = {}
        for v in self.ind_vars:
            src = self.coord_sources.get(v)
            if not src:
                raise KeyError(f"Independent var '{v}' has no coord source mapping.")
            if src not in ds.coords:
                raise KeyError(f"Dataset is missing coord '{src}' required for '{v}'.")
            arr = ds.coords[src].values
            out[v] = self._apply_transform(v, arr, ref)
        return out


@dataclass
class VarMapping:
    """
    Mapping between UPD variable names and PINN dependent variable names.

    Example:
      dep_vars = ["T","U","V"]
      map: {"T":"T2M", "U":"U10M", "V":"V10M"} or direct names.
    """
    dep_vars: List[str]
    upd_to_dep: Dict[str, str]  # dep -> upd_var
    # optional unit conversion hooks (MVP: none)
    enforce_present: bool = True

    def pick_dataset_vars(self, ds: xr.Dataset) -> xr.Dataset:
        """Select dataset variables matching dependent vars; raise if missing and enforce_present."""
        missing = []
        keep = []
        for dep in self.dep_vars:
            upd_name = self.upd_to_dep.get(dep)
            if upd_name is None:
                raise KeyError(f"No mapping for dependent var '{dep}'")
            if upd_name not in ds.data_vars:
                missing.append(upd_name)
            else:
                keep.append(upd_name)

        if missing and self.enforce_present:
            raise KeyError(f"Missing required UPD variables: {missing}")
        return ds[keep]


@dataclass
class PINNMapping:
    """
    Full mapping: coordinates -> independent vars, variables -> dependent vars.
    """
    coord: CoordMapping
    var: VarMapping

    def infer_time0(self, ds: xr.Dataset) -> Optional[np.datetime64]:
        """Infer start time from dataset time coord."""
        if "time" in ds.coords:
            return ds["time"].values.min()
        return None

    def independent_vars(self) -> List[str]:
        """Return list of independent variable names."""
        return list(self.coord.ind_vars)

    def dependent_vars(self) -> List[str]:
        """Return list of dependent variable names."""
        return list(self.var.dep_vars)


def build_default_mapping_atmosphere(
    dep_vars: List[str],
    ind_vars: Optional[List[str]] = None,
    use_lev: bool = False,
    time_transform: str = "seconds_since_start",
    angle_transform: str = "deg2rad",
) -> PINNMapping:
    """
    Default mapping for UPD atmosphere reanalysis shards:
      coords: time, lat, lon, optionally lev
      independent vars:
        - default: ["t","lat","lon"] or ["t","lev","lat","lon"]
      transforms:
        - t: seconds_since_start + minmax to [-1,1]
        - lat/lon: deg2rad + minmax to [-1,1]
    """
    if ind_vars is None:
        ind_vars = ["t", "lat", "lon"] if not use_lev else ["t", "lev", "lat", "lon"]

    coord_sources = {"t": "time", "lat": "lat", "lon": "lon"}
    if use_lev:
        coord_sources["lev"] = "lev"

    coord_transform = {
        "t": time_transform,
        "lat": angle_transform,
        "lon": angle_transform,
    }
    if use_lev:
        coord_transform["lev"] = "identity"

    # By default, assume UPD variable names match dep vars (can override)
    upd_to_dep = {dep: dep for dep in dep_vars}

    return PINNMapping(
        coord=CoordMapping(ind_vars=ind_vars, coord_sources=coord_sources, coord_transform=coord_transform, normalize_to_unit=True),
        var=VarMapping(dep_vars=dep_vars, upd_to_dep=upd_to_dep, enforce_present=True),
    )
