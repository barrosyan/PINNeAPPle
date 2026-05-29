"""Arena gallery — programmatic configs for all 21 physics problems.

Run a single problem:
    python 05_all_problems_gallery.py --problem burgers_1d
    python 05_all_problems_gallery.py --problem allen_cahn --epochs 5000

List available problems:
    python 05_all_problems_gallery.py --list

Run the full gallery (long!):
    python 05_all_problems_gallery.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pinneapple_arena import Arena, ArenaConfig

# ── per-problem config dicts ──────────────────────────────────────────────────

CONFIGS: dict[str, dict] = {

    # ── Fluid Mechanics ───────────────────────────────────────────────────────
    "kovasznay_ns": {
        "problem": {"name": "kovasznay_ns", "params": {"re": 40.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 128, "n_message_passing": 6},
             "training": {"epochs": 800, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/kovasznay_ns/", "prefix": "ns", "save_figures": True},
    },

    "stokes_2d": {
        "problem": {"name": "stokes_2d", "params": {"mu": 0.01}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 8000, "lr": 5e-4}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 128, "n_message_passing": 6},
             "training": {"epochs": 1000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/stokes_2d/", "prefix": "stokes", "save_figures": True},
    },

    # ── Diffusion / Heat ──────────────────────────────────────────────────────
    "heat_1d": {
        "problem": {"name": "heat_1d", "params": {"alpha": 0.01, "t_end": 1.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/heat_1d/", "prefix": "heat1d", "save_figures": True},
    },

    "heat_2d": {
        "problem": {"name": "heat_2d", "params": {"alpha": 0.01, "t_end": 1.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
            {"name": "ModifiedMLP", "type": "modified_mlp",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/heat_2d/", "prefix": "heat2d", "save_figures": True},
    },

    "convection_diffusion_1d": {
        "problem": {"name": "convection_diffusion_1d",
                    "params": {"c": 1.0, "nu": 0.01, "t_end": 1.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/convection_diffusion_1d/", "prefix": "conv_diff",
                   "save_figures": True},
    },

    "darcy_flow_2d": {
        "problem": {"name": "darcy_flow_2d", "params": {}},
        "models": [
            {"name": "FNO-2D", "type": "fno2d",
             "network": {"modes": 12, "width": 32, "layers": 4},
             "training": {"epochs": 600, "lr": 1e-3}},
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 128, "n_message_passing": 6},
             "training": {"epochs": 800, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/darcy_flow_2d/", "prefix": "darcy", "save_figures": True},
    },

    # ── Elliptic ──────────────────────────────────────────────────────────────
    "laplace_2d": {
        "problem": {"name": "laplace_2d", "params": {}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 3000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 3000, "lr": 5e-4}},
            {"name": "ModifiedMLP", "type": "modified_mlp",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 3000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/laplace_2d/", "prefix": "laplace", "save_figures": True},
    },

    "poisson_2d": {
        "problem": {"name": "poisson_2d", "params": {"k": 1.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "FNO-2D", "type": "fno2d",
             "network": {"modes": 12, "width": 32, "layers": 4},
             "training": {"epochs": 500, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/poisson_2d/", "prefix": "poisson", "save_figures": True},
    },

    "helmholtz_2d": {
        "problem": {"name": "helmholtz_2d", "params": {"k": 4.0}},
        "models": [
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 5000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/helmholtz_2d/", "prefix": "helmholtz", "save_figures": True},
    },

    "biharmonic_2d": {
        "problem": {"name": "biharmonic_2d", "params": {}},
        "models": [
            {"name": "SIREN-Deep", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 8000, "lr": 2e-4}},
            {"name": "VanillaPINN-Deep", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/biharmonic_2d/", "prefix": "biharmonic",
                   "save_figures": True},
    },

    # ── Transport ─────────────────────────────────────────────────────────────
    "advection_diffusion_1d": {
        "problem": {"name": "advection_diffusion_1d",
                    "params": {"v": 1.0, "D": 0.05, "t_end": 2.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/advection_diffusion_1d/", "prefix": "adv_diff",
                   "save_figures": True},
    },

    # ── Waves ─────────────────────────────────────────────────────────────────
    "wave_1d": {
        "problem": {"name": "wave_1d", "params": {"c": 1.0, "t_end": 2.0}},
        "models": [
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 8000, "lr": 5e-4}},
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/wave_1d/", "prefix": "wave1d", "save_figures": True},
    },

    "klein_gordon_1d": {
        "problem": {"name": "klein_gordon_1d", "params": {"m": 1.0, "c": 1.0, "t_end": 2.0}},
        "models": [
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 8000, "lr": 5e-4}},
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 8000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/klein_gordon_1d/", "prefix": "klein_gordon",
                   "save_figures": True},
    },

    "nls_1d": {
        "problem": {"name": "nls_1d",
                    "params": {"n_collocation": 20000, "n_ic": 512, "t_end": 1.5708}},
        "models": [
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 15000, "lr": 5e-4}},
            {"name": "VanillaPINN-Deep", "type": "vanilla_pinn",
             "network": {"hidden": [200, 200, 200, 200, 200]},
             "training": {"epochs": 15000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/nls_1d/", "prefix": "nls", "save_figures": True},
    },

    # ── Nonlinear / Solitons ──────────────────────────────────────────────────
    "burgers_1d": {
        "problem": {"name": "burgers_1d",
                    "params": {"nu": 0.01, "n_collocation": 10000, "t_end": 1.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [20, 20, 20, 20, 20, 20, 20, 20]},
             "training": {"epochs": 15000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 10000, "lr": 5e-4}},
            {"name": "ModifiedMLP", "type": "modified_mlp",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 10000, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/burgers_1d/", "prefix": "burgers", "save_figures": True},
    },

    "kdv_1d": {
        "problem": {"name": "kdv_1d",
                    "params": {"n_collocation": 10000, "n_ic": 400, "t_end": 1.0}},
        "models": [
            {"name": "SIREN-High", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 10000, "lr": 5e-4}},
            {"name": "VanillaPINN-Deep", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128, 128]},
             "training": {"epochs": 10000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/kdv_1d/", "prefix": "kdv", "save_figures": True},
    },

    # ── Reaction-Diffusion ────────────────────────────────────────────────────
    "allen_cahn": {
        "problem": {"name": "allen_cahn",
                    "params": {"alpha": 0.0001, "n_collocation": 20000, "n_ic": 512,
                               "t_end": 1.0}},
        "models": [
            {"name": "SIREN-Deep", "type": "siren",
             "network": {"hidden": [256, 256, 256, 256], "omega_0": 30.0},
             "training": {"epochs": 20000, "lr": 5e-4}},
            {"name": "VanillaPINN-Wide", "type": "vanilla_pinn",
             "network": {"hidden": [256, 256, 256, 256]},
             "training": {"epochs": 20000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/allen_cahn/", "prefix": "ac", "save_figures": True},
    },

    "fisher_kpp_1d": {
        "problem": {"name": "fisher_kpp_1d",
                    "params": {"D": 0.01, "r": 1.0, "t_end": 5.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 8000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 8000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/fisher_kpp_1d/", "prefix": "fisher", "save_figures": True},
    },

    "fitzhugh_nagumo_1d": {
        "problem": {"name": "fitzhugh_nagumo_1d",
                    "params": {"D": 0.1, "epsilon": 0.08, "a": 0.7, "b": 0.8,
                               "I_ext": 0.5, "t_end": 20.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 10000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
             "training": {"epochs": 10000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/fitzhugh_nagumo_1d/", "prefix": "fhn", "save_figures": True},
    },

    # ── Structural ────────────────────────────────────────────────────────────
    "linear_elasticity_2d": {
        "problem": {"name": "linear_elasticity_2d", "params": {"E": 200e9, "nu": 0.3}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128]},
             "training": {"epochs": 10000, "lr": 1e-3}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 128, "n_message_passing": 8},
             "training": {"epochs": 1500, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/linear_elasticity_2d/", "prefix": "elasticity",
                   "save_figures": True},
    },

    # ── Finance ───────────────────────────────────────────────────────────────
    "black_scholes_1d": {
        "problem": {"name": "black_scholes_1d",
                    "params": {"sigma": 0.25, "r": 0.05, "K": 100.0, "T": 1.0,
                               "S_min": 20.0, "S_max": 200.0}},
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64]},
             "training": {"epochs": 5000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 5000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/black_scholes_1d/", "prefix": "bs", "save_figures": True},
    },
}


def _patch_epochs(cfg_dict: dict, epochs: int) -> dict:
    """Override epochs in all models without mutating the original."""
    import copy
    d = copy.deepcopy(cfg_dict)
    for m in d.get("models", []):
        m.setdefault("training", {})["epochs"] = epochs
    return d


def run_problem(name: str, epochs: int | None = None) -> None:
    if name not in CONFIGS:
        print(f"Unknown problem '{name}'. Use --list to see available problems.")
        sys.exit(1)

    cfg_dict = CONFIGS[name]
    if epochs is not None:
        cfg_dict = _patch_epochs(cfg_dict, epochs)

    print(f"\n{'='*60}")
    print(f"  Arena: {name}")
    print(f"{'='*60}\n")

    arena = Arena(ArenaConfig.from_dict(cfg_dict))
    arena.run()


def list_all() -> None:
    from pinneapple_arena import list_problems_by_domain
    print("\nAvailable problems by domain:")
    for domain, names in list_problems_by_domain().items():
        print(f"\n  {domain}:")
        for n in names:
            star = " *" if n in CONFIGS else ""
            print(f"    - {n}{star}")
    print("\n(* = has programmatic config in this script)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arena physics problem gallery")
    parser.add_argument("--problem", type=str, help="Problem name to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all models (quick test)")
    parser.add_argument("--list", action="store_true", help="List all problems and exit")
    parser.add_argument("--all", action="store_true",
                        help="Run all 21 problems sequentially (long!)")
    args = parser.parse_args()

    if args.list:
        list_all()
        sys.exit(0)

    if args.all:
        for name in CONFIGS:
            run_problem(name, epochs=args.epochs)
        sys.exit(0)

    if args.problem:
        run_problem(args.problem, epochs=args.epochs)
    else:
        parser.print_help()
        sys.exit(1)
