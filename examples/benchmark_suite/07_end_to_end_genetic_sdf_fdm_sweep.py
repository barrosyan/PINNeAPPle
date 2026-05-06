
from __future__ import annotations

"""End-to-end Arena example (genetic SDF + real solver + multi-GPU sweep).

What this does
--------------
1) Builds a *real genetic geometry* as an implicit SDF (union of circles) and
   extracts boundary points via marching squares.
2) Runs a *real solver* (FDM heat2D) on a bounding-box grid; masks samples to the SDF interior.
3) Normalizes everything into Arena/PINN batch format:
     x_col, x_bc, y_bc, x_ic, y_ic, x_data, y_data, ctx
4) Sweeps **5 models** in parallel (GPU queue), optionally with DDP per model.
5) Saves predictions for all runs and produces a comparison table + ranking + plots.

Run:
  python examples/arena/06_end_to_end_genetic_sdf_fdm_sweep.py
"""

import json
from pathlib import Path

import torch

from pinneaple_arena.runner.run_sweep import run_sweep
from pinneaple_arena.runner.compare import compare_runs


def main():
    # ---------------------------
    # Genetic geometry (serializable config)
    # ---------------------------
    geometry_cfg = {
        "kind": "union_circles_sdf",
        "seed": 7,
        "n_circles": 4,
        "bounds_min": [0.0, 0.0],
        "bounds_max": [1.0, 1.0],
        "r_min": 0.10,
        "r_max": 0.26,
        "intersect_box": True,
    }

    # ---------------------------
    # Physics problem spec (serializable config)
    # Heat equation: u_t - alpha * Laplacian(u) = q (q=0 by default)
    # BC/IC are handled by dataset (y_bc/y_ic); compiler will enforce them.
    # ---------------------------
    alpha = 0.02
    problem_spec_cfg = {
        "name": "heat2d_genetic_sdf",
        "coords": ["x", "y", "t"],
        "fields": ["u"],
        "alpha": alpha,
        "w_bc": 1.0,
        "w_ic": 1.0,
    }

    # ---------------------------
    # Real solver config (FDM heat2d)
    # ---------------------------
    solver_cfg = {
        "solver": "fdm",
        "equation": "heat2d",
        "alpha": alpha,
        "H": 160,
        "W": 160,
        "steps": 240,
        "dt": 7.5e-4,
        # dataset sizes (you can crank these up)
        "n_collocation": 8192,
        "n_boundary": 4096,
        "n_ic": 4096,
        "n_data": 8192,
        "boundary_resolution": 320,
        "seed": 123,
    }

    # ---------------------------
    # Base run config (shared)
    # ---------------------------
    base_run_cfg = {
        "backend": {"name": "pinneaple_models"},
        "arena": {
            "data_source": "solver",
            "geometry_cfg": geometry_cfg,
            "problem_spec_cfg": problem_spec_cfg,
            "solver_cfg": solver_cfg,
            # weights for combined loss
            "loss_weights": {"supervised": 1.0, "physics": 1.0},
            "supervised_kind": "mse",
        },
        "train": {
            "batch_size": 1024,
            "max_steps": 2500,
            "log_every": 100,
            "optimizer": {"name": "adam", "lr": 2e-3},
            "amp": False,
            # DDP is enabled automatically if ddp_per_model=True in run_sweep and torchrun is used
            "ddp": False,
        },
        "model": {"kwargs": {"in_dim": 3, "out_dim": 1}},
    }

    # ---------------------------
    # 5 models in the sweep (all pointwise coords -> scalar u)
    # These are registered by pinneaple_models.benchmarks.registry via register_all().
    # ---------------------------
    models = [
        {"name": "bench_linear", "model": {"kwargs": {"in_dim": 3, "out_dim": 1}}},
        {"name": "bench_mlp", "model": {"kwargs": {"in_dim": 3, "out_dim": 1, "width": 128, "depth": 4, "act": "tanh"}}},
        {"name": "bench_res_mlp", "model": {"kwargs": {"in_dim": 3, "out_dim": 1, "width": 160, "depth": 6}}},
        {"name": "bench_fourier_mlp", "model": {"kwargs": {"in_dim": 3, "out_dim": 1, "num_f": 96, "scale": 12.0, "width": 160, "depth": 4}}},
        {"name": "bench_siren", "model": {"kwargs": {"in_dim": 3, "out_dim": 1, "width": 128, "depth": 4, "w0": 30.0}}},
    ]

    out_dir = "runs/examples_genetic_sweep/heat2d"

    # ---------------------------
    # Multi-GPU sweep strategy
    # - If you have >= 2 GPUs, demonstrate DDP-per-model for the physics-capable models
    # - Otherwise, it will run on CPU/GPU sequentially depending on availability.
    # ---------------------------
    n_gpu = int(torch.cuda.device_count())
    ddp_per_model = bool(n_gpu >= 2)

    run_dirs = run_sweep(
        models=models,
        base_run_cfg=base_run_cfg,
        bundle_root="",  # not used because data_source='solver'
        out_dir=out_dir,
        parallelism="process",
        gpus="auto",
        ddp_per_model=ddp_per_model,
        ddp_world_size=2,
    )

    # ---------------------------
    # Compare all runs + ranking + plots
    # ---------------------------
    comp = compare_runs(
        run_dirs,
        out_dir="runs/examples_genetic_sweep/compare/heat2d",
        title="heat2d_genetic_sdf_fdm_sweep",
        make_plots=True,
    )

    print("\n=== DONE ===")
    print("Runs:", [str(p) for p in run_dirs])
    print("Compare summary:", json.dumps(comp.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
