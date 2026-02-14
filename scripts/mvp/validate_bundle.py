from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pinneaple_arena.bundle.schema import load_bundle_schema
from pinneaple_arena.bundle.loader import load_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="configs/data/bundle_schema.yaml")
    ap.add_argument("--bundle-root", default="data/bundles/flow_obstacle_2d/v0")
    args = ap.parse_args()

    schema = load_bundle_schema(args.schema)
    bundle_root = Path(args.bundle_root)

    # Try USD first; if not present, user likely uses STL and must edit schema accordingly.
    schema.validate_bundle_root(bundle_root)

    b = load_bundle(bundle_root, schema=schema, require_sensors=False)

    # basic sanity checks
    assert b.points_boundary is not None and len(b.points_boundary) > 0
    assert b.points_collocation is not None and len(b.points_collocation) > 0
    assert "region" in b.points_boundary.columns

    regions = set(b.points_boundary["region"].astype(str).unique().tolist())
    expected = {"inlet", "outlet", "walls", "obstacle"}
    missing = sorted(list(expected - regions))
    if missing:
        raise RuntimeError(f"Missing boundary regions: {missing}")

    print("[OK] Bundle validated.")
    print(f" - collocation points: {len(b.points_collocation)}")
    print(f" - boundary points:    {len(b.points_boundary)}")
    print(f" - regions:           {sorted(list(regions))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
