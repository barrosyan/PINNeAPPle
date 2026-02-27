"""05 — Small sweep + leaderboard ranking.

This shows how to treat the arena as an experimentation harness:
- Run multiple configurations
- Append rows to leaderboard.json
- Load and rank results

Run from repo root:
    python examples/arena/05_sweep_and_leaderboard.py
"""

from __future__ import annotations

from pathlib import Path

from pinneaple_arena.runner.run_benchmark import run_benchmark
from pinneaple_arena.runner.leaderboard import load_leaderboard


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "examples" / "sweep"

    task_cfg = repo_root / "examples" / "arena" / "configs" / "task_flow_obstacle_2d.yaml"
    schema = repo_root / "configs" / "data" / "bundle_schema.yaml"

    run_cfgs = [
        repo_root / "examples" / "arena" / "configs" / "run_native_fast.yaml",
        repo_root / "examples" / "arena" / "configs" / "run_native_bigger.yaml",
    ]

    for p in run_cfgs:
        print(f"\n=== running: {p.name} ===")
        out = run_benchmark(
            artifacts_dir=artifacts_dir,
            task_cfg_path=task_cfg,
            run_cfg_path=p,
            bundle_schema_path=schema,
        )
        print("run_id:", out["run_id"])
        print("key metrics:", out["summary"]["key_metrics"])

    # Rank leaderboard
    rows = load_leaderboard(artifacts_dir / "leaderboard.json")

    def key(r):
        # Prefer smaller bc_mse; if missing, push to end
        v = r.get("bc_mse")
        try:
            return float(v)
        except Exception:
            return float("inf")

    rows_sorted = sorted(rows, key=key)

    print("\n=== TOP RUNS by bc_mse (lower is better) ===")
    for r in rows_sorted[:5]:
        print(
            f"bc_mse={r.get('bc_mse')} | test_pde_rms={r.get('test_pde_rms')} | "
            f"run_name={r.get('run_name')} | backend={r.get('backend')} | run_id={r.get('run_id')}"
        )

    print("\nLeaderboard:", artifacts_dir / "leaderboard.json")


if __name__ == "__main__":
    main()
