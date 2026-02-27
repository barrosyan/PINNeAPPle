from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from pinneaple_arena.runner.run_benchmark import run_benchmark


def _p(path: str) -> Path:
    return Path(path).expanduser().resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="run_sweep",
        description="Run pinneaple_arena for one task against multiple run configs (backends) and store comparable artifacts.",
    )
    ap.add_argument("--artifacts-dir", default="artifacts", help="Output dir for arena artifacts (default: artifacts)")
    ap.add_argument("--task", required=True, help="Path to task YAML (e.g., configs/arena/tasks/flow_obstacle_2d.yaml)")
    ap.add_argument("--schema", required=True, help="Path to bundle schema YAML (e.g., configs/data/bundle_schema.yaml)")
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run YAML configs (e.g., configs/arena/runs/run_native.yaml ...)",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining configs even if one backend fails.",
    )

    args = ap.parse_args()

    artifacts_dir = _p(args.artifacts_dir)
    task_cfg = _p(args.task)
    schema_cfg = _p(args.schema)
    run_cfgs: List[Path] = [_p(p) for p in args.runs]

    if not task_cfg.exists():
        raise SystemExit(f"Task config not found: {task_cfg}")
    if not schema_cfg.exists():
        raise SystemExit(f"Bundle schema not found: {schema_cfg}")
    for r in run_cfgs:
        if not r.exists():
            raise SystemExit(f"Run config not found: {r}")

    print(f"\n== Arena Sweep ==")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Task cfg:  {task_cfg}")
    print(f"Schema:    {schema_cfg}")
    print(f"Runs:      {len(run_cfgs)}")
    for r in run_cfgs:
        print(f"  - {r}")

    results = []
    failures = []

    for i, run_cfg in enumerate(run_cfgs, start=1):
        print(f"\n--- [{i}/{len(run_cfgs)}] Running: {run_cfg.name} ---")
        try:
            out = run_benchmark(
                artifacts_dir=str(artifacts_dir),
                task_cfg_path=str(task_cfg),
                run_cfg_path=str(run_cfg),
                bundle_schema_path=str(schema_cfg),
            )
            results.append(out)
            print(f"✅ Done: run_id={out.get('run_id')}  run_dir={out.get('run_dir')}")
            km = out.get("summary", {}).get("key_metrics", {})
            print(
                f"   metrics: pde={km.get('test_pde_rms')} div={km.get('test_div_rms')} "
                f"bc={km.get('bc_mse')} l2uv={km.get('test_l2_uv')}"
            )
        except Exception as e:
            failures.append((run_cfg.name, str(e)))
            print(f"❌ Failed: {run_cfg.name}\n{e}")
            if not args.continue_on_error:
                break

    print("\n== Summary ==")
    print(f"Success: {len(results)}")
    print(f"Failed:  {len(failures)}")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f" - {name}: {msg}")

    print("\nNext:")
    print(" - Compare results: python scripts/compare_arena.py")
    print(" - Leaderboard: artifacts/leaderboard.json")


if __name__ == "__main__":
    main()