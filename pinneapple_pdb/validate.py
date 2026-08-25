"""Validation and standardization for physical datasets (UPD)."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass
class ValidationSpec:
    """Specification for dataset validation (units, ranges, monotonicity)."""

    require_units: bool = True
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "T": (150.0, 350.0),    # K
        "Q": (0.0, 0.1),        # kg/kg (tune per product)
        "U": (-200.0, 200.0),   # m/s
        "V": (-200.0, 200.0),   # m/s
        "PS": (5e4, 1.1e5),     # Pa (sea-level/high-altitude surface pressure)
    })
    enforce_lev_monotonic: bool = True
    enforce_time_monotonic: bool = True


# Explicit variable -> physical-quantity mapping for range checks. Only variables
# actually representing temperature/humidity/wind belong here; names that merely
# start with the same letter (TQV, TQL, TROPPB, TO3, ...) must NOT be swept in.
_QUANTITY_BY_NAME: Dict[str, str] = {
    "TS": "T", "T2M": "T", "T2MDEW": "T", "T2MWET": "T",
    "TROPT": "T", "TROPPT": "T",
    "QV2M": "Q", "QV10M": "Q", "TROPQ": "Q",
    "U2M": "U", "U10M": "U", "U50M": "U",
    "V2M": "V", "V10M": "V", "V50M": "V",
    "PS": "PS",
}
# Pressure-level fields like T850/Q500/U250/V850 follow a <LETTER><level> pattern.
_PLEV_RE = re.compile(r"^([TQUV])\d+$")


def _quantity_for(name: str) -> Optional[str]:
    """Map a variable name to a physical quantity tag ("T"/"Q"/"U"/"V"/"PS"), or None."""
    q = _QUANTITY_BY_NAME.get(name)
    if q is not None:
        return q
    m = _PLEV_RE.match(name)
    if m:
        return m.group(1)
    return None


def standardize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dimension names and transpose data vars to canonical order (time, lev, lat, lon)."""
    rename = {}
    for k in list(ds.dims):
        lk = k.lower()
        if lk in ("latitude", "lat"):
            rename[k] = "lat"
        elif lk in ("longitude", "lon"):
            rename[k] = "lon"
        elif lk in ("level", "lev", "plev", "isobaric", "isobaricinhpa"):
            rename[k] = "lev"
        elif lk == "time":
            rename[k] = "time"
    if rename:
        ds = ds.rename(rename)

    def target_order(dims: List[str]) -> List[str]:
        if "lev" in dims:
            order = [d for d in ["time", "lev", "lat", "lon"] if d in dims]
        else:
            order = [d for d in ["time", "lat", "lon"] if d in dims]
        for d in dims:
            if d not in order:
                order.append(d)
        return order

    data_vars = {}
    for name, da in ds.data_vars.items():
        data_vars[name] = da.transpose(*target_order(list(da.dims)))
    return xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)


def _sample_flat(da: xr.DataArray, max_n: int = 200_000):
    """Sample finite values from a DataArray (subsampled if larger than max_n)."""
    x = da.values.ravel()
    if x.size > max_n:
        idx = np.random.choice(x.size, size=max_n, replace=False)
        x = x[idx]
    x = x[np.isfinite(x)]
    return x


def validate_dataset(ds: xr.Dataset, spec: ValidationSpec) -> List[str]:
    """Validate dataset against spec; returns list of issue messages (empty if valid)."""
    issues: List[str] = []

    # Units
    if spec.require_units:
        for name, da in ds.data_vars.items():
            if not str(da.attrs.get("units", "")).strip():
                issues.append(f"[units] '{name}' sem units")

    # Time monotonic
    if spec.enforce_time_monotonic and "time" in ds.coords:
        t = ds["time"].values
        if len(t) > 1:
            ti = t.astype("datetime64[ns]").astype("int64")
            if not np.all(np.diff(ti) > 0):
                issues.append("[time] time não é estritamente crescente")

    # Lev monotonic
    if spec.enforce_lev_monotonic and "lev" in ds.coords:
        lev = ds["lev"].values.astype(float)
        if len(lev) > 1:
            d = np.diff(lev)
            if not (np.all(d > 0) or np.all(d < 0)):
                issues.append("[lev] lev não é monotônico")

    def check_range(name: str, da: xr.DataArray, lo: float, hi: float, tag: str):
        x = _sample_flat(da)
        if x.size == 0:
            return
        if x.min() < lo or x.max() > hi:
            issues.append(
                f"[range:{tag}] '{name}' fora ~({lo},{hi}) min={float(x.min()):.3g} max={float(x.max()):.3g}"
            )

    for name, da in ds.data_vars.items():
        quantity = _quantity_for(name)
        if quantity in ("T", "U", "V"):
            lo, hi = spec.ranges.get(quantity, (-np.inf, np.inf))
            check_range(name, da, lo, hi, quantity)
        elif quantity == "Q":
            lo, hi = spec.ranges.get("Q", (-np.inf, np.inf))
            check_range(name, da, lo, hi, "Q")
            try:
                if float(da.min()) < 0:
                    issues.append(f"[nonneg] '{name}' tem valores negativos")
            except Exception:
                pass
        elif quantity == "PS":
            lo, hi = spec.ranges.get("PS", (-np.inf, np.inf))
            check_range(name, da, lo, hi, "PS")
            try:
                if float(da.min()) <= 0:
                    issues.append("[phys] PS <= 0 encontrado")
            except Exception:
                pass

    return issues
