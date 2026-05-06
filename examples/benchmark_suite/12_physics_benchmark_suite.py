"""12_physics_benchmark_suite.py — PINN Arena Multi-Architecture Benchmark.

Tests 6 neural architectures across 6 physics problems and ranks them.

Problems
--------
  burgers_1d      — Burgers equation 1D (nu=0.01/pi, shock at t~0.3)
  poisson_2d      — Poisson equation 2D (exact sin(pi*x)sin(pi*y))
  heat_1d         — Heat diffusion 1D (exact exp(-pi^2*alpha*t)sin(pi*x))
  wave_1d         — Wave equation 1D (exact cos(c*pi*t)sin(pi*x))
  allen_cahn_1d   — Allen-Cahn phase field 1D (eps=0.01, FDM reference)
  ns_tgv_2d       — Navier-Stokes 2D Taylor-Green Vortex (exact solution)

Architectures
-------------
  VanillaPINN_S   — MLP 4x64 tanh
  VanillaPINN_M   — MLP 6x128 tanh
  VanillaPINN_L   — MLP 8x256 tanh
  ResNetPINN      — Residual network 4 blocks x128
  FourierPINN     — Random Fourier features + MLP 4x128
  SIREN           — Sinusoidal activations 4x128 omega=30

Usage
-----
  # Full benchmark (all 36 runs, ~60 epochs for quick test)
  python 12_physics_benchmark_suite.py

  # Fast mode (fewer epochs, subset of problems/models)
  python 12_physics_benchmark_suite.py --fast

  # Specific problems and models
  python 12_physics_benchmark_suite.py --problems burgers_1d poisson_2d --models VanillaPINN_M SIREN

  # Resume from saved results
  python 12_physics_benchmark_suite.py --load results/benchmark_results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="PINN Arena Benchmark Suite")
    p.add_argument("--fast", action="store_true", help="Quick test mode (100 epochs)")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs per run")
    p.add_argument(
        "--problems",
        nargs="+",
        default=None,
        choices=["burgers_1d", "poisson_2d", "heat_1d", "wave_1d", "allen_cahn_1d", "ns_tgv_2d"],
        help="Problems to benchmark (default: all)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["VanillaPINN_S", "VanillaPINN_M", "VanillaPINN_L", "ResNetPINN", "FourierPINN", "SIREN"],
        help="Models to benchmark (default: all)",
    )
    p.add_argument("--device", default="auto", help="Device: auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save", default="results/benchmark_results.json", help="Output JSON path")
    p.add_argument("--load", default=None, help="Load and display saved results instead of running")
    p.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load-only mode ────────────────────────────────────────────────────────
    if args.load:
        from pinneaple_arena.benchmark import PINNArenaBenchmark
        import json

        print(f"\nLoading results from {args.load}")
        data = PINNArenaBenchmark.load_results(args.load)
        print(f"  {len(data)} entries loaded")
        for entry in sorted(data, key=lambda r: (r["problem_id"], r.get("metrics", {}).get("rel_l2", 9e9))):
            m = entry.get("metrics", {})
            print(
                f"  {entry['problem_id']:<18} {entry['model_id']:<20} "
                f"rel_l2={m.get('rel_l2', float('nan')):.3e}  "
                f"t={entry.get('elapsed_s', 0):.1f}s"
            )
        return

    # ── Build benchmark ───────────────────────────────────────────────────────
    from pinneaple_arena.benchmark import PINNArenaBenchmark, BenchmarkConfig

    epochs = args.epochs or (100 if args.fast else 5000)

    print("\n" + "=" * 70)
    print("  PINN Arena — Physics Benchmark Suite")
    print(f"  Mode     : {'FAST (100 ep)' if args.fast else 'FULL'}")
    print(f"  Epochs   : {epochs}")
    print(f"  Problems : {args.problems or 'all (6)'}")
    print(f"  Models   : {args.models or 'all (6)'}")
    print(f"  Device   : {args.device}")
    print("=" * 70)

    # Configure weights — upweight IC/BC for well-posedness
    cfg = BenchmarkConfig(
        epochs=epochs,
        lr=1e-3,
        weight_pde=1.0,
        weight_bc=10.0,
        weight_ic=10.0,
        device=args.device,
        seed=args.seed,
        n_col=4000,
        n_bc=800,
        n_ic=800,
        n_eval=10000,
        convergence_threshold=1e-3,
        log_interval=max(1, epochs // 5),
    )

    bench = PINNArenaBenchmark.default(
        problems=args.problems,
        models=args.models,
        epochs=epochs,
        device=args.device,
    )
    bench.config = cfg

    # ── Run ───────────────────────────────────────────────────────────────────
    results = bench.run(verbose=True)

    # ── Reports ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(bench.leaderboard(by_problem=True))
    print()
    print(bench.leaderboard(by_problem=False))
    print()
    print(bench.summary_table())

    best = bench.best_per_problem()
    print("\n  Best model per problem:")
    for pid, mid in best.items():
        run = next((r for r in results if r.problem_id == pid and r.model_id == mid), None)
        rel_l2 = run.metrics.get("rel_l2", float("nan")) if run else float("nan")
        print(f"    {pid:<18} → {mid:<20} (rel_l2={rel_l2:.3e})")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = Path(args.save)
    bench.save_results(str(save_path))

    # ── Plots (optional) ──────────────────────────────────────────────────────
    if args.plot:
        plot_dir = save_path.parent / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        bench.plot_leaderboard(save_path=str(plot_dir / "leaderboard.png"))
        print(f"  Leaderboard plot saved to {plot_dir / 'leaderboard.png'}")

        for task in bench.tasks:
            bench.plot_convergence(
                problem_id=task.task_id,
                save_path=str(plot_dir / f"convergence_{task.task_id}.png"),
            )
        print(f"  Convergence plots saved to {plot_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
