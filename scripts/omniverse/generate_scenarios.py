"""
RUN INSIDE OMNIVERSE KIT.

Creates N copies of a base USD file (useful for scenario sweep).
This script does NOT modify geometry; it just duplicates the file.
"""

from __future__ import annotations

from pathlib import Path

import omni.usd


def duplicate_usd(base_usd: str, out_dir: str, n: int = 10) -> None:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ctx = omni.usd.get_context()

    for i in range(int(n)):
        ctx.open_stage(base_usd)
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("Could not open stage in Omniverse context.")

        out_path = out_dir_p / f"scenario_{i:04d}.usd"
        stage.GetRootLayer().Export(str(out_path))
        print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    duplicate_usd(
        base_usd="C:/cenas/flow_obstacle_2d.usd",
        out_dir="C:/PINNeAPPle/data/cache/omniverse_exports",
        n=5,
    )
