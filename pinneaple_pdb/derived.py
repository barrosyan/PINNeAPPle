"""Derived variable computation (e.g. vorticity, divergence) for physical datasets."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

EARTH_R = 6_371_000.0  # m

@dataclass
class DerivedSpec:
    """Specification for derived variables to add (e.g. vorticity, divergence)."""

    derived: List[str] = field(default_factory=list)  # ["vorticity", "divergence"]
    u_name: Optional[str] = None
    v_name: Optional[str] = None

def _infer_uv(
    ds: xr.Dataset, u_name: Optional[str], v_name: Optional[str]
) -> Tuple[str, str]:
    """Infer U/V variable names from dataset (prefers U850/V850, etc.)."""
    if u_name and v_name:
        if u_name not in ds.data_vars or v_name not in ds.data_vars:
            raise ValueError("u_name/v_name não encontrados no dataset.")
        return u_name, v_name

    candidates_u = [n for n in ds.data_vars if n.startswith("U")]
    candidates_v = [n for n in ds.data_vars if n.startswith("V")]
    if not candidates_u or not candidates_v:
        raise ValueError("Não encontrei variáveis U* e V* para derivadas.")

    for lvl in ["850", "500", "250"]:
        uu, vv = f"U{lvl}", f"V{lvl}"
        if uu in ds.data_vars and vv in ds.data_vars:
            return uu, vv
    return candidates_u[0], candidates_v[0]

def add_vorticity_divergence(
    ds: xr.Dataset, u_name: Optional[str] = None, v_name: Optional[str] = None
) -> xr.Dataset:
    """Add vorticity and divergence from U/V on lat/lon grid; returns new dataset."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Precisa de coords lat/lon para vorticidade/divergência.")

    uvar, vvar = _infer_uv(ds, u_name, v_name)
    U = ds[uvar]
    V = ds[vvar]

    lat = ds["lat"].values.astype(float)
    lon = ds["lon"].values.astype(float)

    # radians spacing (assumes regular-ish grids)
    dlat = np.deg2rad(np.gradient(lat))
    dlon = np.deg2rad(np.gradient(lon))

    coslat = np.cos(np.deg2rad(lat))
    coslat = np.clip(coslat, 1e-6, None)

    def grad_lon(A):
        return np.gradient(A, axis=-1) / dlon  # per rad
    def grad_lat(A):
        return np.gradient(A, axis=-2) / dlat  # per rad

    u = U.values
    v = V.values

    dv_dlon = grad_lon(v)
    du_dlon = grad_lon(u)
    dv_dlat = grad_lat(v)
    du_dlat = grad_lat(u)

    # broadcast cos(lat) to match (..., lat, lon)
    if u.ndim >= 3:
        cos_b = coslat[None, ..., None]
    else:
        cos_b = coslat[..., None]

    dv_dx = dv_dlon / (EARTH_R * cos_b)
    du_dx = du_dlon / (EARTH_R * cos_b)
    dv_dy = dv_dlat / EARTH_R
    du_dy = du_dlat / EARTH_R

    zeta = dv_dx - du_dy
    div = du_dx + dv_dy

    zeta_da = xr.DataArray(zeta, dims=U.dims, coords=U.coords, name="vorticity")
    div_da  = xr.DataArray(div,  dims=U.dims, coords=U.coords, name="divergence")

    zeta_da.attrs.update({"units": "1/s", "derived_from": f"{uvar},{vvar}"})
    div_da.attrs.update({"units": "1/s", "derived_from": f"{uvar},{vvar}"})

    return ds.assign({"vorticity": zeta_da, "divergence": div_da})

def apply_derived(ds: xr.Dataset, spec: DerivedSpec) -> xr.Dataset:
    """Apply derived variables from spec to dataset; returns new dataset."""
    want = set([s.lower().strip() for s in (spec.derived or [])])
    if not want:
        return ds
    if ("vorticity" in want) or ("divergence" in want):
        ds = add_vorticity_divergence(ds, u_name=spec.u_name, v_name=spec.v_name)
    return ds
