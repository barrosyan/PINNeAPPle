"""07_easy_custom_problem.py — High-level custom problem API (define_problem).

This example shows how to define and solve physics problems using the
``define_problem()`` function — no subclassing, no manual autograd, no data
generation boilerplate.

You only provide:
  - equation name and description
  - coordinates and their domain bounds
  - field names (unknowns)
  - PDE as a plain string (e.g. ``"eps * u_xx - k * u + f"``)
  - boundary / initial conditions as simple dicts
  - (optional) analytical solution for error metrics

The system automatically:
  - parses derivative notation (``u_xx``, ``u_t``, ``v_xy``, …) into autograd calls
  - generates physics and BC losses
  - samples interior / boundary / evaluation points
  - wires up the full Arena training pipeline
  - saves figures

Derivative notation reference
------------------------------
  u_x   → ∂u/∂x       u_xx  → ∂²u/∂x²
  u_t   → ∂u/∂t       v_xy  → ∂²v/∂x∂y
  p_yy  → ∂²p/∂y²     etc.

Usage
-----
    python examples/arena_pipelines/07_easy_custom_problem.py
    python examples/arena_pipelines/07_easy_custom_problem.py --case 1d
    python examples/arena_pipelines/07_easy_custom_problem.py --case 2d
    python examples/arena_pipelines/07_easy_custom_problem.py --case heat
    python examples/arena_pipelines/07_easy_custom_problem.py --epochs 2000
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneaple_arena import define_problem


# ══════════════════════════════════════════════════════════════════════════════
# Case 1 — 1-D Reaction-Diffusion
# ══════════════════════════════════════════════════════════════════════════════
# PDE: eps·u'' - k·u = -(eps·π² + k)·sin(πx)  on [0,1]
# BC:  u(0) = u(1) = 0
# Solution: u(x) = sin(πx)

def case_reaction_diffusion_1d(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 1: 1-D Reaction-Diffusion (define_problem)")
    print("=" * 60)

    prob = define_problem(
        name="rd1d_easy",
        description="eps·u'' − k·u = f  on [0,1],  u(0)=u(1)=0",

        # Coordinate name → (min, max)
        coords={"x": (0.0, 1.0)},

        # Unknown field(s) to solve for
        fields=["u"],

        # PDE as a string (residual = 0).
        # Derivative notation: u_xx = ∂²u/∂x²
        # Param names (eps, k) and coord names (x) are in scope.
        pde="eps * u_xx - k * u + (eps * pi**2 + k) * sin(pi * x)",

        # Boundary conditions
        bcs=[
            {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
        ],

        # Physical parameters
        params={"eps": 0.1, "k": 1.0},

        # Known solution for L2/Linf error metrics
        analytical=lambda x, eps=0.1, k=1.0, **kw: {"u": np.sin(np.pi * x)},
    )

    # One-liner: train, evaluate, visualize
    prob.solve(
        models=["VanillaPINN", "SIREN"],
        epochs=epochs,
        output_dir="outputs/easy_custom/rd1d/",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 2 — 2-D Poisson
# ══════════════════════════════════════════════════════════════════════════════
# PDE: Δu = -(2π²)·sin(πx)·sin(πy)   on [0,1]²
# BC:  u = 0  on all edges
# Solution: u(x,y) = sin(πx)·sin(πy)

def case_poisson_2d(epochs: int = 4000):
    print("\n" + "=" * 60)
    print("  Case 2: 2-D Poisson (define_problem)")
    print("=" * 60)

    prob = define_problem(
        name="poisson2d_easy",
        description="Δu = f  on [0,1]²,  u=0 on ∂Ω",

        coords={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        fields=["u"],

        # The source term is embedded directly in the PDE string
        pde="u_xx + u_yy + 2 * pi**2 * sin(pi * x) * sin(pi * y)",

        # Single BC covering the entire boundary
        bcs=[
            {"type": "dirichlet", "at": "boundary", "field": "u", "value": 0.0},
        ],

        params={},
        analytical=lambda x, y, **kw: {"u": np.sin(np.pi * x) * np.sin(np.pi * y)},
    )

    prob.solve(
        models=["VanillaPINN", "SIREN"],
        epochs=epochs,
        output_dir="outputs/easy_custom/poisson2d/",
        grid_n=30,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 3 — 1-D+time Heat Equation
# ══════════════════════════════════════════════════════════════════════════════
# PDE: u_t = alpha·u_xx   on [0,1] × [0,1]
# BC:  u(0,t) = u(1,t) = 0
# IC:  u(x,0) = sin(πx)
# Solution: u(x,t) = exp(-alpha·π²·t)·sin(πx)

def case_heat_1d(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 3: 1-D Heat Equation (define_problem)")
    print("=" * 60)

    alpha = 0.1

    prob = define_problem(
        name="heat1d_easy",
        description="u_t = alpha·u_xx  on [0,1]×[0,1]",

        coords={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        fields=["u"],

        # Residual: u_t - alpha·u_xx = 0
        pde="u_t - alpha * u_xx",

        bcs=[
            # Spatial boundaries (Dirichlet, all t)
            {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
            # Initial condition: u(x, 0) = sin(πx)
            # Value is a callable f(x, t, **params) → array
            {
                "type": "initial",
                "at": "t_min",
                "field": "u",
                "value": lambda x, t, **kw: np.sin(np.pi * x),
            },
        ],

        params={"alpha": alpha},
        analytical=lambda x, t, alpha=0.1, **kw: {
            "u": np.exp(-alpha * math.pi**2 * t) * np.sin(math.pi * x)
        },
    )

    prob.solve(
        models=["VanillaPINN", "ModifiedMLP"],
        epochs=epochs,
        output_dir="outputs/easy_custom/heat1d/",
        grid_n=30,
        n_col=1500,
        n_bc=300,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Case 4 — Helmholtz equation (advanced: callable BC value, string expr)
# ══════════════════════════════════════════════════════════════════════════════
# PDE: u_xx + u_yy + k²·u = f   on [0,1]²
# BC:  u = sin(πx)·sin(πy)  on ∂Ω  (non-zero, matches the solution)
# Solution: u = sin(πx)·sin(πy)

def case_helmholtz_2d(epochs: int = 4000):
    print("\n" + "=" * 60)
    print("  Case 4: 2-D Helmholtz (advanced BCs, define_problem)")
    print("=" * 60)

    k = 1.0
    source = f"({k**2} - 2 * pi**2) * sin(pi * x) * sin(pi * y)"

    prob = define_problem(
        name="helmholtz2d_easy",
        description="u_xx + u_yy + k²u = f  on [0,1]²",

        coords={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        fields=["u"],

        pde=f"u_xx + u_yy + k**2 * u - ({source})",

        bcs=[
            # BC value as callable — matches the exact solution on each edge
            {
                "type": "dirichlet",
                "at": "boundary",
                "field": "u",
                "value": lambda x, y, **kw: np.sin(np.pi * x) * np.sin(np.pi * y),
            },
        ],

        params={"k": k},
        analytical=lambda x, y, k=1.0, **kw: {
            "u": np.sin(np.pi * x) * np.sin(np.pi * y)
        },
    )

    prob.solve(
        models=["VanillaPINN", "SIREN"],
        epochs=epochs,
        output_dir="outputs/easy_custom/helmholtz2d/",
        grid_n=30,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="define_problem() examples")
    parser.add_argument(
        "--case",
        choices=["1d", "2d", "heat", "helmholtz", "all"],
        default="all",
        help="Which case to run (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all cases")
    args = parser.parse_args()

    defaults = {"1d": 3000, "2d": 4000, "heat": 3000, "helmholtz": 4000}

    def _ep(case):
        return args.epochs if args.epochs is not None else defaults[case]

    cases = {
        "1d":        lambda: case_reaction_diffusion_1d(_ep("1d")),
        "2d":        lambda: case_poisson_2d(_ep("2d")),
        "heat":      lambda: case_heat_1d(_ep("heat")),
        "helmholtz": lambda: case_helmholtz_2d(_ep("helmholtz")),
    }

    if args.case == "all":
        for fn in cases.values():
            fn()
    else:
        cases[args.case]()


if __name__ == "__main__":
    main()
