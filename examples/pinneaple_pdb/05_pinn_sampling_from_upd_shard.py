"""Sample PINN-ready collocation/condition/data points from a UPD shard.

This uses pinneaple_pinn.io.UPDDataset and a PINNMapping that maps:
  inputs:  t, lat, lon
  targets: one UPD variable (e.g. T2M)

Run after you have a catalog and at least one shard, e.g. from:
  - 03_offline_synthetic_upd_pipeline.py

Requirements:
  pip install pandas pyarrow xarray zarr torch numpy
"""

from __future__ import annotations

import os
import pandas as pd

from pinneaple_pinn.io.upd_dataset import UPDItem, UPDDataset, SamplingSpec, ConditionSpec
from pinneaple_pinn.io.mappings import CoordMapping, VarMapping, PINNMapping


def main():
    catalog_path = "examples/_out/pinneaple_pdb/synthetic_build/catalog.parquet"
    if not os.path.exists(catalog_path):
        raise FileNotFoundError("Run 03_offline_synthetic_upd_pipeline.py first.")

    df = pd.read_parquet(catalog_path).sort_values("uid")
    row = df.iloc[0].to_dict()

    item = UPDItem(zarr_path=row["zarr_path"], meta_path=row["meta_path"])

    # Map coords -> independent vars
    coord = CoordMapping(
        ind_vars=["t", "lat", "lon"],
        coord_sources={"t": "time", "lat": "lat", "lon": "lon"},
        coord_transform={"t": "seconds_since_start", "lat": "deg2rad", "lon": "deg2rad"},
        normalize_to_unit=True,
    )

    # Map one UPD var -> dependent var
    var = VarMapping(dep_vars=["T"], var_sources={"T": "T2M"}, normalize_to_unit=True)

    mapping = PINNMapping(coord=coord, var=var)

    ds = UPDDataset(item=item, mapping=mapping, device="cpu")

    spec = SamplingSpec(
        n_collocation=2048,
        conditions=[
            ConditionSpec(name="t0", type="initial", equation="T(t0,lat,lon)=T_data", n=512),
            ConditionSpec(name="bnd", type="boundary", equation="periodic/edges", n=512),
        ],
        n_data=512,
        seed=0,
    )

    batch = ds.sample(spec)

    print("collocation:", [x.shape for x in batch.collocation] if batch.collocation else None)
    if batch.conditions:
        for i, c in enumerate(batch.conditions):
            print(f"condition[{i}]:", [x.shape for x in c])
    if batch.data:
        xin, y = batch.data
        print("data x:", [x.shape for x in xin], "y:", y.shape)


if __name__ == "__main__":
    main()
