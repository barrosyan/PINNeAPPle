from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/artifacts/leaderboard.json")
    ap.add_argument("--sort", default="test_pde_rms", help="Metric key to sort by (ascending)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print("No leaderboard.json found.")
        return 0

    rows = json.loads(p.read_text(encoding="utf-8"))
    key = args.sort

    def k(r):
        v = r.get(key, None)
        try:
            return float(v)
        except Exception:
            return float("inf")

    rows = sorted(rows, key=k)

    print(f"=== Leaderboard (sorted by {key}) ===")
    for r in rows[: int(args.top)]:
        name = r.get("run_name", "n/a")
        rid = r.get("run_id", "n/a")
        pde = r.get("test_pde_rms", None)
        div = r.get("test_div_rms", None)
        bc = r.get("bc_mse", None)
        print(f"- {name:28s} run_id={rid}  pde={pde}  div={div}  bc={bc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
