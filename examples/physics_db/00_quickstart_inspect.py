"""Quickstart: discover a dataset via CMR/earthaccess and inspect variables.

This demonstrates the *discovery* workflow of pinneaple_pdb:

  1) login() (NASA Earthdata)
  2) list_collections() to find dataset short_name/provider
  3) set_dataset() + set_spacetime()
  4) inspect() opens the first granule and returns:
     - dims/coords
     - variables + units
     - suggested variable packs (core_state_2d, upper_air_plev, ...)

Requirements (online):
  - pip install earthaccess cmr xarray pydap netcdf4
  - NASA Earthdata account (earthaccess.login will prompt)

Tip:
  If you only want to *see* what's available, start with list_collections()
  using a keyword like "MERRA" or "GEOS".
"""

from __future__ import annotations

import json
from pprint import pprint

from pinneaple_pdb import PhysicalDatasetBuilder


def main():
    b = PhysicalDatasetBuilder()

    # 1) Search collections (keep keyword broad at first)
    collections = b.list_collections(keyword="MERRA", limit=10)
    print("\nTop matches:")
    for c in collections[:10]:
        print(f"- {c.get('short_name')} | provider={c.get('provider')} | {c.get('title','')[:80]}")

    # 2) Pick one (edit as needed)
    # Many users end up using MERRA-2 products; pick a short_name shown above.
    b.set_dataset(provider=collections[0].get("provider"), short_name=collections[0].get("short_name"))

    # 3) Spacetime window (keep small for inspection)
    b.set_spacetime(
        time_start="2020-01-01",
        time_end="2020-01-02",
        bbox=(-45.0, -15.0, -30.0, -5.0),  # (W,S,E,N) roughly NE Brazil
    )

    # 4) Inspect one granule
    info = b.inspect(prefer="opendap", engine="pydap", max_granules=2)

    print("\nPicked URL:")
    print(info["picked_url"])

    print("\nDims:")
    pprint(info["dims"])

    print("\nFirst 20 variables:")
    for v in info["variables"][:20]:
        print(f"- {v['name']:12s} dims={v['dims']} units={v['units']}")

    print("\nSuggested packs:")
    pprint(info["suggested_packs"])

    # Save to file for later browsing
    out = "examples/_out/pinneaple_pdb"
    import os
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/inspect.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out}/inspect.json")


if __name__ == "__main__":
    main()
