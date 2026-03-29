"""09 — End-to-end pipeline from a YAML config.

Runs the complete workflow in a single call:
  geometry → solver (reference data) → dataset → train 3 models → inference → report

What this demonstrates
----------------------
- pinneaple_arena.run_full_pipeline() as single entry point
- Reading all settings from a YAML config file
- Multi-model training with comparison metrics
- Automatic report generation (loss curves, field plots, error maps)
- GPU-ready config (switch device: cuda + amp: true for GPU)

Run from repo root:
    python examples/pinneaple_arena/09_full_pipeline_yaml.py

Or point to any custom YAML:
    python examples/pinneaple_arena/09_full_pipeline_yaml.py \\
        --config examples/pinneaple_arena/configs/pipeline_burgers_full.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pinneaple_arena import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="pinneaple full pipeline")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "examples" / "pinneaple_arena" / "configs" / "pipeline_burgers_full.yaml"),
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (overrides config.pipeline.out_dir)",
    )
    args = parser.parse_args()

    print(f"[pinneaple] Running full pipeline from: {args.config}")
    print(f"[pinneaple] Output dir: {args.out_dir or '(from config)'}")
    print()

    summary = run_full_pipeline(args.config, out_dir=args.out_dir)

    # Print results summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Run name : {summary.get('name', 'N/A')}")
    print(f"  Out dir  : {summary.get('out_dir', 'N/A')}")
    print(f"  Status   : {summary.get('status', 'N/A')}")

    timing = summary.get("timing", {})
    if timing:
        print(f"\n  Step timings:")
        for step, secs in timing.items():
            print(f"    {step:<20} {secs:.1f}s")

    model_results = summary.get("model_results", {})
    if model_results:
        print(f"\n  Model results ({len(model_results)} models):")
        for model_id, res in model_results.items():
            metrics = res.get("metrics", {})
            val_loss = res.get("val_loss", float("nan"))
            print(f"    [{model_id}]  val_loss={val_loss:.4e}", end="")
            for k in ("rel_l2", "rmse", "r2"):
                if k in metrics:
                    print(f"  {k}={metrics[k]:.4f}", end="")
            print()

    report_path = summary.get("report_path")
    if report_path:
        print(f"\n  Report   : {report_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
