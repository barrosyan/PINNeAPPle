from __future__ import annotations

import argparse

from pinneaple_arena.runner.run_benchmark import run_benchmark


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="data/artifacts")
    ap.add_argument("--task", default="configs/arena/tasks/flow_obstacle_2d.yaml")
    ap.add_argument("--schema", default="configs/data/bundle_schema.yaml")
    ap.add_argument("--run", required=True, help="Run yaml path (e.g. configs/arena/runs/vanilla_pinn_native.yaml)")
    args = ap.parse_args()

    res = run_benchmark(
        artifacts_dir=args.artifacts,
        task_cfg_path=args.task,
        run_cfg_path=args.run,
        bundle_schema_path=args.schema,
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
