"""09_dataset_benchmarks.py — Arena benchmarks on real pinneaple_data datasets.

This example shows three complementary APIs for building Arena benchmarks
from datasets already integrated with PINNeAPPle:

  1. **benchmark_dataset()** — single-function, one-liner for any preset dataset.
  2. **DatasetProblem.from_dataset()** — fine-grained control, plugs into any
     existing Arena config.
  3. **Physics-hybrid mode** (``mode="pinn_data"``) — combines a real dataset
     with an autograd PDE residual for improved generalisation.
  4. **list_benchmarks()** — discover all available presets.

Supported problem classes
--------------------------
  PDE             burgers_1d, heat_1d, wave_1d, poisson_2d, helmholtz_2d,
                  allen_cahn_1d, kovasznay_ns, navier_stokes_2d …
  Timeseries      lorenz63, spring_mass, van_der_pol, double_pendulum,
                  lotka_volterra …
  Geometry/CFD    naca0012, cylinder_2d, channel_flow, lid_driven_cavity …
  Regression      airfoil_noise, concrete_strength, energy_efficiency …
  Inverse         burgers_1d_inverse, heat_1d_inverse …

Usage
-----
    python examples/arena_pipelines/09_dataset_benchmarks.py
    python examples/arena_pipelines/09_dataset_benchmarks.py --case burgers
    python examples/arena_pipelines/09_dataset_benchmarks.py --case kovasznay
    python examples/arena_pipelines/09_dataset_benchmarks.py --case hybrid
    python examples/arena_pipelines/09_dataset_benchmarks.py --case timeseries
    python examples/arena_pipelines/09_dataset_benchmarks.py --case regression
    python examples/arena_pipelines/09_dataset_benchmarks.py --case list
    python examples/arena_pipelines/09_dataset_benchmarks.py --case all
    python examples/arena_pipelines/09_dataset_benchmarks.py --epochs 1000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneaple_arena import (
    Arena,
    ArenaConfig,
    DatasetProblem,
    benchmark_dataset,
    list_benchmarks,
    get_benchmark_preset,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: show the preset table
# ══════════════════════════════════════════════════════════════════════════════

def case_list():
    """Print every available dataset benchmark preset."""
    print("\n" + "=" * 62)
    print("  Case: list_benchmarks()")
    print("=" * 62)
    list_benchmarks()

    print("\nFilter by category:")
    list_benchmarks(category="pde", verbose=True)


# ══════════════════════════════════════════════════════════════════════════════
# Case 1 — Burgers 1-D (supervised, one-liner)
# ══════════════════════════════════════════════════════════════════════════════

def case_burgers(epochs: int = 3000):
    """Viscous Burgers equation — one-liner benchmark from a real dataset."""
    print("\n" + "=" * 62)
    print("  Case 1: Burgers 1-D (benchmark_dataset, supervised)")
    print("=" * 62)

    # All defaults come from the 'burgers_1d' preset.
    # Pass epochs to override the preset default.
    benchmark_dataset(
        "burgers_1d",
        models=["VanillaPINN", "SIREN"],
        epochs=epochs,
        mode="supervised",
        output_dir="outputs/dataset_bench/",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 2 — Kovasznay NS (DatasetProblem + manual Arena config)
# ══════════════════════════════════════════════════════════════════════════════

def case_kovasznay(epochs: int = 3000):
    """Kovasznay NS — use DatasetProblem directly in a hand-crafted Arena config."""
    print("\n" + "=" * 62)
    print("  Case 2: Kovasznay NS (DatasetProblem.from_dataset + Arena)")
    print("=" * 62)

    # Create the problem manually — gives full control over field mapping
    prob = DatasetProblem.from_dataset(
        "kovasznay_ns",
        input_fields=["x", "y"],
        output_fields=["u", "v", "p"],
        mode="supervised",
        name="kovasznay_manual",
        description="Kovasznay 2-D NS steady flow benchmark (Re=40)",
        re=40.0,
    )
    print(f"  Problem created: {prob}")

    # Build a custom Arena config referencing the problem by name
    cfg = ArenaConfig.from_dict({
        "problem": {"name": prob.name},
        "models": [
            {
                "name": "VanillaPINN",
                "type": "vanilla_pinn",
                "network": {"hidden": [128, 128, 128, 128]},
                "training": {"epochs": epochs, "lr": 1e-3},
            },
            {
                "name": "SIREN",
                "type": "siren",
                "network": {"hidden": [128, 128, 128, 128]},
                "training": {"epochs": epochs, "lr": 5e-4},
            },
        ],
        "output": {
            "dir": "outputs/dataset_bench/kovasznay_manual/",
            "prefix": "kovasznay",
        },
    })

    Arena(cfg).run(n_col=3000, n_bc=400, grid_n=40)


# ══════════════════════════════════════════════════════════════════════════════
# Case 3 — Physics-hybrid mode (burgers pinn_data)
# ══════════════════════════════════════════════════════════════════════════════

def case_hybrid(epochs: int = 4000):
    """Burgers 1-D physics-hybrid: real data + autograd PDE residual.

    Physics-hybrid mode combines:
      * Supervised loss: MSE against the dataset solution
      * Physics residual: autograd Burgers residual u_t + u·u_x − ν·u_xx = 0

    This often achieves better generalisation than pure supervised training,
    especially with limited data.
    """
    print("\n" + "=" * 62)
    print("  Case 3: Burgers 1-D physics-hybrid (pinn_data mode)")
    print("=" * 62)

    # mode="pinn_data" activates the built-in Burgers residual
    benchmark_dataset(
        "burgers_1d",
        models=["VanillaPINN", "ModifiedMLP"],
        epochs=epochs,
        mode="pinn_data",          # <── hybrid physics + data
        n_train=2000,
        n_bc=300,
        output_dir="outputs/dataset_bench/",
        Nx=256,
        Nt=101,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 4 — Timeseries: Lorenz-63 attractor
# ══════════════════════════════════════════════════════════════════════════════

def case_timeseries(epochs: int = 3000):
    """Lorenz-63 chaotic ODE — time→state regression from dataset."""
    print("\n" + "=" * 62)
    print("  Case 4: Lorenz-63 timeseries benchmark")
    print("=" * 62)

    benchmark_dataset(
        "lorenz63",
        models=["VanillaPINN", "ModifiedMLP"],
        epochs=epochs,
        mode="timeseries",
        n_train=2000,
        output_dir="outputs/dataset_bench/",
        sigma=10.0,
        rho=28.0,
        beta=2.667,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 5 — Regression: airfoil noise
# ══════════════════════════════════════════════════════════════════════════════

def case_regression(epochs: int = 2000):
    """UCI Airfoil Self-Noise regression — 5 features → sound pressure level."""
    print("\n" + "=" * 62)
    print("  Case 5: Airfoil noise regression benchmark")
    print("=" * 62)

    benchmark_dataset(
        "airfoil_noise",
        models=["VanillaPINN", "ModifiedMLP"],
        epochs=epochs,
        mode="supervised",
        n_train=1000,
        output_dir="outputs/dataset_bench/",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 6 — Get and inspect a preset
# ══════════════════════════════════════════════════════════════════════════════

def case_preset_inspect():
    """Show how to read and customise a preset before running."""
    print("\n" + "=" * 62)
    print("  Case 6: Inspect and customise a benchmark preset")
    print("=" * 62)

    preset = get_benchmark_preset("allen_cahn_1d")
    print(f"\n  Preset for 'allen_cahn_1d':")
    for k, v in preset.items():
        print(f"    {k:<20} = {v}")

    # Mutate a copy and run with custom overrides
    print("\n  Running Allen–Cahn with doubled epochs and FourierPINN …")
    benchmark_dataset(
        "allen_cahn_1d",
        models=["VanillaPINN", "FourierPINN"],
        epochs=max(preset["epochs"] // 4, 500),   # quick run for demo
        mode="pinn_data",
        output_dir="outputs/dataset_bench/",
        eps=0.01,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 7 — Heat 1-D with physics-hybrid (pinn_data)
# ══════════════════════════════════════════════════════════════════════════════

def case_heat_hybrid(epochs: int = 3000):
    """Heat equation physics-hybrid: dataset + autograd u_t = k·u_xx residual."""
    print("\n" + "=" * 62)
    print("  Case 7: Heat 1-D physics-hybrid (pinn_data)")
    print("=" * 62)

    benchmark_dataset(
        "heat_1d",
        models=["VanillaPINN", "SIREN"],
        epochs=epochs,
        mode="pinn_data",
        n_train=2000,
        n_bc=300,
        output_dir="outputs/dataset_bench/",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

CASE_MAP = {
    "list":        lambda ep: case_list(),
    "burgers":     case_burgers,
    "kovasznay":   case_kovasznay,
    "hybrid":      case_hybrid,
    "timeseries":  case_timeseries,
    "regression":  case_regression,
    "preset":      lambda ep: case_preset_inspect(),
    "heat":        case_heat_hybrid,
}

CASE_DEFAULTS = {
    "burgers":    3000,
    "kovasznay":  3000,
    "hybrid":     4000,
    "timeseries": 3000,
    "regression": 2000,
    "heat":       3000,
}


def main():
    parser = argparse.ArgumentParser(
        description="Arena benchmarks from real pinneaple_data datasets"
    )
    parser.add_argument(
        "--case",
        choices=list(CASE_MAP.keys()) + ["all"],
        default="all",
        help="Which case to run (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs for all cases",
    )
    args = parser.parse_args()

    def _ep(case):
        return args.epochs if args.epochs is not None else CASE_DEFAULTS.get(case, 2000)

    if args.case == "all":
        for name, fn in CASE_MAP.items():
            fn(_ep(name))
    else:
        CASE_MAP[args.case](_ep(args.case))


if __name__ == "__main__":
    main()
