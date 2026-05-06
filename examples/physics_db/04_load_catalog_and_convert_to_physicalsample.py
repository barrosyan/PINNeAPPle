"""Load a UPD catalog, open one shard, and convert it to PhysicalSample.

This bridges:
  pinneaple_pdb (writes shard zarr+json+catalog)
    -> pinneaple_data.adapters.upd_adapter (converts to PhysicalSample)

Run after:
  - 03_offline_synthetic_upd_pipeline.py  (offline)
    or
  - 02_build_upd_earthaccess_sharded.py   (online)

Requirements:
  pip install pandas pyarrow xarray zarr numpy torch
"""

from __future__ import annotations

import os
import pandas as pd

from pinneaple_data.adapters.upd_adapter import upd_to_physical_sample


def main():
    # Choose one:
    catalog_path = "examples/_out/pinneaple_pdb/synthetic_build/catalog.parquet"
    # catalog_path = "examples/_out/pinneaple_pdb/earthaccess_build/catalog.parquet"

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}. Run the build example first.")

    df = pd.read_parquet(catalog_path).sort_values("uid")
    row = df.iloc[0].to_dict()

    print("Picked uid:", row["uid"])
    print("zarr_path:", row["zarr_path"])
    print("meta_path:", row["meta_path"])

    sample = upd_to_physical_sample((row["zarr_path"], row["meta_path"]))
    ds = sample.state

    print("\nPhysicalSample fields:")
    print("state type:", type(sample.state))
    print("vars:", list(ds.data_vars)[:10], "...")
    print("coords:", list(ds.coords))
    print("dims:", {k: int(v) for k, v in ds.sizes.items()})
    print("schema keys:", list((sample.schema or {}).keys())[:10])
    print("provenance keys:", list((sample.provenance or {}).keys())[:10])

    # quick numeric check: mean/std of one var
    v0 = list(ds.data_vars)[0]
    arr = ds[v0].values
    import numpy as np
    print(f"\n{v0}: mean={np.nanmean(arr):.4f} std={np.nanstd(arr):.4f}")


if __name__ == "__main__":
    main()
