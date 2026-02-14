"""
RUN INSIDE OMNIVERSE KIT.

Exports a bundle for the MVP. It calls your integration function:
  pinneaple_integrations.omniverse.export_flow_bundle_from_usd

If that integration is not present yet, this script will fail with a clear error.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    usd_path = "C:/cenas/flow_obstacle_2d.usd"
    out_bundle_dir = "C:/PINNeAPPle/data/bundles/flow_obstacle_2d/v0"

    # If you need to add repo path inside Kit, uncomment and set:
    # sys.path.append("C:/PINNeAPPle")

    try:
        from pinneaple_integrations.omniverse import export_flow_bundle_from_usd
    except Exception as e:
        raise RuntimeError(
            "Could not import pinneaple_integrations.omniverse.export_flow_bundle_from_usd.\n"
            "You must implement this integration in your repo under pinneaple_integrations/omniverse.\n"
            f"Import error: {e}"
        )

    export_flow_bundle_from_usd(
        usd_path=usd_path,
        out_bundle_dir=out_bundle_dir,
        domain_xy=((0.0, 2.0), (0.0, 1.0)),
        obstacle_circle=(0.7, 0.5, 0.15),
        nu=0.01,
        n_boundary=20000,
        n_collocation=200000,
        seed=0,
    )

    print(f"[OK] Bundle exported to: {out_bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
