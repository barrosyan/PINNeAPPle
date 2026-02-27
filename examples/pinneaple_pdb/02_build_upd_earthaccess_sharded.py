"""Build UPD shards from an Earthdata collection (online) and write catalog.

This is the "full" pinneaple_pdb pipeline:

  - select variables (by include list and/or packs)
  - standardize dims to (time, lev?, lat, lon)
  - add derived vars (vorticity/divergence)
  - validate (units, ranges, monotonic coords)
  - shard by time window + optional spatial tiling
  - write each shard as:
      <uid>.zarr   (xarray Dataset)
      <uid>.json   (metadata including schema)
    and append to a Parquet catalog

Requirements:
  pip install earthaccess cmr xarray zarr pandas pyarrow pydap netcdf4 numpy

Note:
  You'll need a NASA Earthdata login the first time (earthaccess.login()).
  If you are in a non-interactive environment, configure earthaccess auth
  following their docs (netrc / environment).

"""

from __future__ import annotations

import os
from pinneaple_pdb import (
    PhysicalDatasetBuilder,
    ShardSpec,
    DerivedSpec,
    ValidationSpec,
)


def main():
    out_root = "examples/_out/pinneaple_pdb/earthaccess_build"
    os.makedirs(out_root, exist_ok=True)

    out_dir = os.path.join(out_root, "shards")
    catalog_path = os.path.join(out_root, "catalog.parquet")

    b = PhysicalDatasetBuilder()

    # ---- 1) Dataset
    # Use the quickstart inspect script to find a good provider/short_name.
    # These placeholders are intentionally explicit:
    provider = "GES_DISC"   # edit
    short_name = "M2I6NPANA"  # edit (example-ish; depends on what CMR returns)
    b.set_dataset(provider=provider, short_name=short_name)

    # ---- 2) Space/time region
    b.set_spacetime(
        time_start="2020-01-01",
        time_end="2020-01-03",
        bbox=(-45.0, -15.0, -30.0, -5.0),  # (W,S,E,N) NE Brazil window
        chunking={"time": 4, "lat": 181, "lon": 288},
    )

    # ---- 3) Variable selection (packs are resolved against available vars)
    # You can mix: packs + explicit include; exclude always wins.
    b.set_selection(
        packs=["core_state_2d", "upper_air_plev"],
        include=[],
        exclude=["TO3", "TOX"],  # example: skip chemistry if too heavy
    )

    # ---- 4) Schema (physical metadata stored in each shard's JSON)
    b.set_schema_from_template("atmosphere_reanalysis_v1")

    # ---- 5) Sharding + derived + validation
    shards = ShardSpec(time_window="6H", tile_deg=(30.0, 30.0), add_regime_tags=True)
    derived = DerivedSpec(derived=["vorticity", "divergence"])  # infers U*/V* names
    validate = ValidationSpec(require_units=True)

    # ---- 6) Build
    result = b.build(
        out_dir=out_dir,
        catalog_path=catalog_path,
        shards=shards,
        derived=derived,
        validate=validate,
        prefer="opendap",
        engine="pydap",
        max_granules=2,
    )

    print("\nBuild summary:")
    print("written:", len(result.get("written", [])))
    print("errors:", len(result.get("errors", [])))
    print("catalog:", catalog_path)
    if result.get("errors"):
        print("First error:", result["errors"][0])


if __name__ == "__main__":
    main()
