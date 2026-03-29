"""02 — Arena experiment from YAML config.

Demonstrates the full Arena workflow:
- Load a problem preset from pinneaple_environment (Burgers 1D)
- Generate training data via a built-in FDM solver
- Train multiple MLP-PINN models with different sizes
- Evaluate metrics and produce visualizations

Run from repo root:
    python examples/pinneaple_arena/02_arena_from_yaml.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pinneaple_arena.runner.run_arena_yaml import run_arena_experiment


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "examples" / "pinneaple_arena" / "configs" / "experiment_burgers_1d.yaml"
    out_dir = repo_root / "data" / "artifacts" / "experiments" / "burgers_1d"

    print("=" * 60)
    print("PINNeAPPle Arena — Burgers 1D experiment")
    print("=" * 60)
    print(f"Config : {config_path}")
    print(f"Out dir: {out_dir}")
    print()

    result = run_arena_experiment(config_path, out_dir=out_dir)

    print("\n=== RESULTS ===")
    summary = result["summary"]
    for model_id, mdata in summary["models"].items():
        print(f"\nModel: {model_id}")
        print(f"  best_val     : {mdata['best_val']:.4g}")
        print(f"  elapsed (s)  : {mdata['elapsed_sec']:.1f}")
        if mdata["eval_metrics"]:
            for k, v in mdata["eval_metrics"].items():
                print(f"  {k:12s}: {v:.4g}")

    if summary.get("visualizations"):
        print("\nSaved plots:")
        for name, path in summary["visualizations"].items():
            print(f"  {name}: {path}")

    print(f"\nFull results at: {result['out_dir']}")


if __name__ == "__main__":
    main()
