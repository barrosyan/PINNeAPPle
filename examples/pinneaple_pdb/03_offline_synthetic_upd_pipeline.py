"""Offline demo: exercise pinneaple_pdb sharding/derived/validation on a synthetic xarray Dataset.

This script is runnable without Earthdata credentials and shows the *core mechanics*
behind the builder:

  - create synthetic (time, lat, lon) winds U/V
  - standardize dims
  - apply derived variables (vorticity/divergence)
  - validate ranges/units
  - shard by time windows and spatial tiles
  - write UPD shard zarr + json + parquet catalog (same on-disk contract as builder.build)

Requirements:
  pip install xarray zarr pandas pyarrow numpy

"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pinneaple_pdb.builder import PhysicalDatasetBuilder
from pinneaple_pdb.shard import ShardSpec, iter_time_windows, subset_time, iter_tiles, subset_tile, regime_tags_for
from pinneaple_pdb.validate import ValidationSpec, standardize_dims, validate_dataset
from pinneaple_pdb.derived import DerivedSpec, apply_derived


def make_synthetic_ds() -> xr.Dataset:
    # coords
    time = pd.date_range("2020-01-01", periods=16, freq="3H")
    lat = np.linspace(-20, 0, 41)
    lon = np.linspace(-50, -30, 81)

    # simple rotating flow field
    tt = np.arange(len(time), dtype=np.float64)[:, None, None]
    yy = lat[None, :, None]
    xx = lon[None, None, :]

    U10M = np.sin(np.deg2rad(yy)) * np.cos(0.1 * tt)
    V10M = -np.cos(np.deg2rad(xx)) * np.sin(0.1 * tt)

    # add another scalar field
    T2M = 300.0 + 2.0 * np.sin(np.deg2rad(yy)) + 0.5 * np.cos(np.deg2rad(xx))

    ds = xr.Dataset(
        data_vars={
            "U10M": (("time", "lat", "lon"), U10M.astype(np.float32), {"units": "m s-1"}),
            "V10M": (("time", "lat", "lon"), V10M.astype(np.float32), {"units": "m s-1"}),
            "T2M":  (("time", "lat", "lon"), T2M.astype(np.float32),  {"units": "K"}),
        },
        coords={
            "time": time.values,
            "lat": lat.astype(np.float32),
            "lon": lon.astype(np.float32),
        },
        attrs={"title": "synthetic_demo"},
    )
    return ds


def main():
    out_root = Path("examples/_out/pinneaple_pdb/synthetic_build")
    out_dir = out_root / "shards"
    catalog_path = out_root / "catalog.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # mimic what builder.build does after opening a dataset
    ds = make_synthetic_ds()
    ds = standardize_dims(ds)

    # derived
    ds = apply_derived(ds, DerivedSpec(derived=["vorticity", "divergence"], u_name="U10M", v_name="V10M"))

    # validation
    validate = ValidationSpec(require_units=True)
    validate_dataset(ds, validate)

    # shard spec
    shards = ShardSpec(time_window="6H", tile_deg=(10.0, 10.0), add_regime_tags=True)

    # We reuse builder._write_upd so the format stays identical.
    b = PhysicalDatasetBuilder()
    b.hub.provider = "offline"
    b.hub.short_name = "synthetic"
    b.spacetime.time_start = "2020-01-01"
    b.spacetime.time_end = "2020-01-03"
    b.selection.include = list(ds.data_vars)

    notes = {"offline": True, "why": "demo"}

    written = []
    for t0, t1 in iter_time_windows(ds["time"].values, shards.time_window):
        dts = subset_time(ds, t0, t1)
        if shards.tile_deg is None:
            tags = regime_tags_for(dts) if shards.add_regime_tags else []
            b._write_upd(dts, out_dir, str(catalog_path), url="offline://synthetic", notes=notes, t0=t0, t1=t1, tags=tags, tile=None)
            written.append((str(t0), str(t1), None))
        else:
            dlat, dlon = shards.tile_deg
            for tile in iter_tiles(dts["lat"].values, dts["lon"].values, dlat, dlon):
                dtt = subset_tile(dts, tile)
                if int(dtt.sizes.get("lat", 0)) == 0 or int(dtt.sizes.get("lon", 0)) == 0:
                    continue
                tags = regime_tags_for(dtt) if shards.add_regime_tags else []
                b._write_upd(dtt, out_dir, str(catalog_path), url="offline://synthetic", notes=notes, t0=t0, t1=t1, tags=tags, tile=tile)
                written.append((str(t0), str(t1), tile))

    print(f"Wrote {len(written)} shards")
    print("Catalog:", catalog_path)
    print("Shard dir:", out_dir)

    # Show first catalog rows
    import pandas as pd
    df = pd.read_parquet(catalog_path)
    print(df.head(5))


if __name__ == "__main__":
    main()
