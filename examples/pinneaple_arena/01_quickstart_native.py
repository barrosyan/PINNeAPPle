"""01 — Quickstart: run the built-in FlowObstacle2D task with the native backend.

What this demonstrates:
- Bundle loading + schema validation
- Backend training
- Task metric computation (PDE residuals + BC + optional sensors)
- Saving run artifacts + updating leaderboard

Run from repo root:
    python examples/arena/01_quickstart_native.py
"""

from __future__ import annotations

from pathlib import Path

from pinneaple_arena.runner.run_benchmark import run_benchmark


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "examples" / "quickstart_native"

    out = run_benchmark(
        artifacts_dir=artifacts_dir,
        task_cfg_path=repo_root / "examples" / "pinneaple_arena" / "configs" / "task_flow_obstacle_2d.yaml",
        run_cfg_path=repo_root / "examples" / "pinneaple_arena" / "configs" / "run_native_fast.yaml",
        bundle_schema_path=repo_root / "configs" / "data" / "bundle_schema.yaml",
    )

    print("\n=== DONE ===")
    print("run_id:", out["run_id"])
    print("run_dir:", out["run_dir"])
    print("summary:")
    for k, v in out["summary"]["key_metrics"].items():
        print(f"  - {k}: {v}")
    print("leaderboard:", artifacts_dir / "leaderboard.json")


if __name__ == "__main__":
    main()
