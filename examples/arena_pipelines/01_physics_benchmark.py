"""
01_physics_benchmark.py — PhysicsBenchmarkSpec examples

Demonstrates:
  1. Quick single-model run (Burgers 1D, VanillaPINN)
  2. Multi-model comparison (Heat 1D — VanillaPINN vs SIREN vs ModifiedMLP)
  3. Inverse problem (Heat 1D — recover diffusivity k)
  4. Kovasznay Navier-Stokes benchmark

Run with:
    cd examples/arena_pipelines
    python 01_physics_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneaple_tools.benchmark_suite import PhysicsBenchmarkSpec

OUT = Path("outputs") / "physics"

# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — Quick single-model run: Burgers 1D
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 1 — Burgers 1D  (VanillaPINN, 1500 epochs)")
print("="*60)

spec = PhysicsBenchmarkSpec(
    problem            = "burgers_1d",
    load_generate_data = "generate",
    source             = None,             # auto-load reference from datasets
    metrics            = ["mse", "l2_rel", "max_err"],
    collocation_points = "sobol",
    models             = ["vanilla_pinn"],
    inverse            = False,
    inverse_variables  = [],
    plots              = True,
    epochs             = 1500,
    lr                 = 1e-3,
    n_col              = 2000,
    n_bc               = 300,
    n_ic               = 300,
    hidden             = [64, 64, 64],
    seed               = 42,
    output_dir         = str(OUT),
)

report1 = spec.run()
report1.save(OUT / "burgers_1d_report.json")
print(f"  Report saved → {OUT / 'burgers_1d_report.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — Multi-model comparison: Heat 1D
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 2 — Heat 1D  (VanillaPINN vs SIREN, 2000 epochs)")
print("="*60)

spec2 = PhysicsBenchmarkSpec(
    problem            = "heat_1d",
    load_generate_data = "generate",
    metrics            = ["mse", "l2_rel"],
    collocation_points = "lhs",
    models             = ["vanilla_pinn", "siren"],
    inverse            = False,
    plots              = True,
    epochs             = 2000,
    lr                 = 1e-3,
    n_col              = 2000,
    n_bc               = 300,
    n_ic               = 300,
    hidden             = [64, 64, 64, 64],
    seed               = 0,
    output_dir         = str(OUT),
)

report2 = spec2.run()
report2.save(OUT / "heat_1d_comparison_report.json")

print(f"\n  Best model: {report2.best_model}")
print(f"  Leaderboard: {report2.leaderboard}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — Inverse problem: recover thermal diffusivity k
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 3 — Inverse Heat 1D  (recover k, true=0.40)")
print("="*60)

spec3 = PhysicsBenchmarkSpec(
    problem            = "heat_1d",
    load_generate_data = "load",           # load reference data as observations
    source             = "heat_1d",        # dataset ID
    metrics            = ["mse", "l2_rel"],
    collocation_points = "sobol",
    models             = ["vanilla_pinn"],
    inverse            = True,
    inverse_variables  = ["k"],            # recover diffusivity
    plots              = True,
    epochs             = 2500,
    lr                 = 1e-3,
    n_col              = 2000,
    n_bc               = 300,
    n_ic               = 300,
    hidden             = [64, 64, 64, 64],
    seed               = 7,
    output_dir         = str(OUT),
)

report3 = spec3.run()
report3.save(OUT / "heat_1d_inverse_report.json")

if report3.model_results.get("vanilla_pinn"):
    p_est = report3.model_results["vanilla_pinn"].param_estimates
    print(f"\n  Identified k = {p_est}")
    print(f"  True k = 0.40")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — Kovasznay Navier-Stokes  (steady 2D NS, analytical reference)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 4 — Kovasznay NS  (2D steady, VanillaPINN)")
print("="*60)

spec4 = PhysicsBenchmarkSpec(
    problem            = "kovasznay_ns",
    load_generate_data = "generate",
    metrics            = ["mse", "l2_rel"],
    collocation_points = "sobol",
    models             = ["vanilla_pinn"],
    plots              = True,
    epochs             = 3000,
    lr                 = 1e-3,
    n_col              = 3000,
    n_bc               = 500,
    n_ic               = 0,
    hidden             = [64, 64, 64, 64],
    seed               = 42,
    output_dir         = str(OUT),
)

report4 = spec4.run()
report4.save(OUT / "kovasznay_ns_report.json")

print("\n" + "="*60)
print("  All physics benchmarks complete.")
print(f"  Results saved to: {OUT.resolve()}")
print("="*60)
