from __future__ import annotations

import argparse
from pathlib import Path
import json

from pinneaple_arena.runner.run_benchmark import run_benchmark


RUNS = [
    "configs/arena/runs/vanilla_pinn_native.yaml",
    "configs/arena/runs/physicsnemo_sym.yaml",
    "configs/arena/runs/physicsnemo_sym_large.yaml",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="data/artifacts")
    ap.add_argument("--task", default="configs/arena/tasks/flow_obstacle_2d.yaml")
    ap.add_argument("--schema", default="configs/data/bundle_schema.yaml")
    ap.add_argument("--only", default="", help="Comma-separated subset of run yaml paths")
    args = ap.parse_args()

    runs = RUNS
    if args.only.strip():
        pick = [s.strip() for s in args.only.split(",") if s.strip()]
        runs = pick

    results = []
    for r in runs:
        print(f"\n=== Running: {r} ===")
        out = run_benchmark(
            artifacts_dir=args.artifacts,
            task_cfg_path=args.task,
            run_cfg_path=r,
            bundle_schema_path=args.schema,
        )
        results.append(out)

    out_path = Path(args.artifacts) / "reports" / "matrix_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Wrote matrix summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
