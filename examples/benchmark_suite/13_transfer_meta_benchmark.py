"""13_transfer_meta_benchmark.py — Transfer Learning & Meta-Learning Benchmark.

Compares three learning paradigms on parametric PDE families:

  1. PROBLEM-SPECIFIC (scratch): train a dedicated model from scratch for each task
  2. TRANSFER LEARNING: pre-train on a source task, then fine-tune on targets
     using different freezing strategies (finetune / partial_freeze / feature_extract)
  3. META-LEARNING: meta-train on a PDE family (MAML or Reptile), then K-shot adapt

Key question: how quickly can each approach reach accurate solutions on a new,
related physics problem — given a fixed adaptation budget of K gradient steps?

Usage
-----
  # Full benchmark (all 3 paradigms, default settings)
  python 13_transfer_meta_benchmark.py

  # Fast mode for quick testing
  python 13_transfer_meta_benchmark.py --fast

  # Only transfer learning, heat scenario
  python 13_transfer_meta_benchmark.py --mode transfer --scenarios heat_alpha

  # Only meta-learning, wave family
  python 13_transfer_meta_benchmark.py --mode meta --families wave_c

  # Save plots
  python 13_transfer_meta_benchmark.py --fast --plot --save results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Transfer & Meta-Learning Benchmark")
    p.add_argument("--mode", default="both", choices=["transfer", "meta", "both"])
    p.add_argument("--fast", action="store_true", help="Quick test (few epochs)")
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        choices=["burgers_nu", "heat_alpha", "wave_c"],
        help="Transfer scenarios (default: all)",
    )
    p.add_argument(
        "--families", nargs="+", default=None,
        choices=["burgers_nu", "heat_alpha", "wave_c"],
        help="Meta-learning families (default: all)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--save", default=None, help="Directory to save results + plots")
    p.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Transfer learning benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_benchmark(args):
    from pinneaple_arena.transfer_benchmark import TransferBenchmarkPipeline, TransferBenchmarkConfig

    if args.fast:
        cfg_overrides = dict(n_source_epochs=100, n_finetune_epochs=50)
    else:
        cfg_overrides = dict(n_source_epochs=3000, n_finetune_epochs=1000)

    print("\n" + "=" * 72)
    print("  TRANSFER LEARNING BENCHMARK")
    print("=" * 72)

    pipe = TransferBenchmarkPipeline.default(
        scenarios=args.scenarios,
        epochs_source=cfg_overrides["n_source_epochs"],
        epochs_finetune=cfg_overrides["n_finetune_epochs"],
        device=args.device,
    )

    results = pipe.run(verbose=True)

    print("\n" + "=" * 72)
    print("  TRANSFER RESULTS")
    print("=" * 72)
    print(pipe.leaderboard())

    # Save
    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe.save_results(str(out_dir / "transfer_results.json"))

        if args.plot:
            for scene in pipe.scenarios:
                pipe.plot_comparison(
                    scenario_name=scene.name,
                    save_path=str(out_dir / f"transfer_compare_{scene.name}.png"),
                )
                for tgt_label in scene.target_labels:
                    pipe.plot_convergence(
                        scenario_name=scene.name,
                        target_label=tgt_label,
                        save_path=str(out_dir / f"transfer_convergence_{scene.name}_{tgt_label[:10]}.png"),
                    )
            print(f"  Plots saved to {out_dir}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learning benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_meta_benchmark(args):
    from pinneaple_arena.meta_benchmark import MetaBenchmarkPipeline, MetaBenchmarkConfig

    if args.fast:
        n_meta_epochs = 30
        k_shots = (1, 5, 10, 20)
    else:
        n_meta_epochs = 500
        k_shots = (1, 5, 10, 20, 50, 100, 200)

    print("\n" + "=" * 72)
    print("  META-LEARNING BENCHMARK")
    print("=" * 72)

    pipe = MetaBenchmarkPipeline.default(
        families=args.families,
        n_meta_epochs=n_meta_epochs,
        device=args.device,
    )
    pipe.config.k_shots = k_shots

    results = pipe.run(verbose=True)

    print("\n" + "=" * 72)
    print("  META-LEARNING RESULTS")
    print("=" * 72)

    for fam in pipe.families:
        print(pipe.summary_k_shot(family_name=fam.name))
        print()

    max_k = max(r.k_shot for r in results if r.algorithm not in ("scratch_full",))
    print(pipe.leaderboard(k_shot=max_k))

    # Save
    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe.save_results(str(out_dir / "meta_results.json"))

        if args.plot:
            for fam in pipe.families:
                pipe.plot_k_shot_curves(
                    family_name=fam.name,
                    save_path=str(out_dir / f"meta_kshot_{fam.name}.png"),
                )
                pipe.plot_meta_convergence(
                    family_name=fam.name,
                    save_path=str(out_dir / f"meta_convergence_{fam.name}.png"),
                )
            print(f"  Plots saved to {out_dir}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Combined comparison summary
# ─────────────────────────────────────────────────────────────────────────────

def print_combined_summary(transfer_results, meta_results):
    """Print a unified comparison across all paradigms."""
    print("\n" + "=" * 72)
    print("  COMBINED COMPARISON SUMMARY")
    print("  (After full training budget for each paradigm)")
    print("=" * 72)

    if transfer_results:
        import math
        from pinneaple_arena.transfer_benchmark import TransferBenchmarkResult

        best_transfer: dict = {}
        for r in transfer_results:
            key = (r.scenario_name, r.target_label)
            val = r.metrics.get("rel_l2", float("nan"))
            if not math.isnan(val):
                if key not in best_transfer or val < best_transfer[key][1]:
                    best_transfer[key] = (r.strategy, val)

        print("\n  Transfer Learning (best strategy per target):")
        for (scene, tgt), (strat, val) in best_transfer.items():
            print(f"    {scene} -> {tgt:<28}  best={strat:<22}  rel_l2={val:.3e}")

    if meta_results:
        import math, numpy as np
        from pinneaple_arena.meta_benchmark import MetaBenchmarkResult

        max_k = max((r.k_shot for r in meta_results if r.algorithm not in ("scratch_full",)), default=0)

        print(f"\n  Meta-Learning at K={max_k} (avg over eval tasks):")
        families = list(dict.fromkeys(r.family_name for r in meta_results))
        algos = list(dict.fromkeys(r.algorithm for r in meta_results
                                   if r.algorithm != "scratch_full"))
        for fam in families:
            print(f"    Family: {fam}")
            for algo in algos:
                vals = [r.metrics.get("rel_l2", float("nan"))
                        for r in meta_results
                        if r.family_name == fam and r.algorithm == algo and r.k_shot == max_k
                        and not math.isnan(r.metrics.get("rel_l2", float("nan")))]
                avg = float(np.mean(vals)) if vals else float("nan")
                print(f"      {algo:<16}  avg_rel_l2={avg:.3e}")

        # Oracle
        print("\n  Full-scratch oracle (same total steps as meta-training):")
        for fam in families:
            vals = [r.metrics.get("rel_l2", float("nan"))
                    for r in meta_results
                    if r.family_name == fam and r.algorithm == "scratch_full"
                    and not math.isnan(r.metrics.get("rel_l2", float("nan")))]
            import numpy as np
            avg = float(np.mean(vals)) if vals else float("nan")
            print(f"    {fam:<20}  oracle_rel_l2={avg:.3e}")

    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    transfer_results = None
    meta_results = None

    if args.mode in ("transfer", "both"):
        transfer_results = run_transfer_benchmark(args)

    if args.mode in ("meta", "both"):
        meta_results = run_meta_benchmark(args)

    if transfer_results or meta_results:
        print_combined_summary(transfer_results, meta_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
