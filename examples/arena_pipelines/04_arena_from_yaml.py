"""04_arena_from_yaml.py — run the Arena benchmark from a YAML config.

Usage
-----
    python examples/arena_pipelines/04_arena_from_yaml.py
    python examples/arena_pipelines/04_arena_from_yaml.py --config path/to/config.yaml
    python examples/arena_pipelines/04_arena_from_yaml.py --problem burgers_1d
"""
import argparse
import os
import sys

# ── project root on path ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneaple_arena import Arena, ArenaConfig


# ── pre-built configs for common problems ─────────────────────────────────────

BUILTIN_CONFIGS = {
    "kovasznay_ns": {
        "problem": {
            "name": "kovasznay_ns",
            "params": {"re": 40.0},
            "grid_n": 40, "n_col": 2000, "n_bc": 500,
            "n_train_supervised": 200, "n_mesh_nodes": 600,
        },
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [128, 128, 128, 128], "activation": "tanh"},
             "training": {"epochs": 2000, "lr": 5e-4, "grad_clip": 1.0}},
            {"name": "FNO-2D", "type": "fno2d",
             "network": {"width": 32, "modes": 12, "layers": 4},
             "training": {"epochs": 600, "lr": 1e-3}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 128, "n_message_passing": 6},
             "training": {"epochs": 600, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/kovasznay", "prefix": "benchmark",
                   "save_figures": True, "dpi": 150, "dark_theme": True},
    },
    "burgers_1d": {
        "problem": {
            "name": "burgers_1d",
            "params": {"nu": 0.01},
            "grid_n": 40, "n_col": 2000, "n_bc": 400,
            "n_train_supervised": 200,
        },
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64, 64], "activation": "tanh"},
             "training": {"epochs": 3000, "lr": 1e-3}},
            {"name": "SIREN", "type": "siren",
             "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
             "training": {"epochs": 3000, "lr": 5e-4}},
        ],
        "output": {"dir": "outputs/burgers", "prefix": "benchmark",
                   "save_figures": True, "dpi": 150, "dark_theme": True},
    },
    "poisson_2d": {
        "problem": {
            "name": "poisson_2d",
            "params": {},
            "grid_n": 40, "n_col": 1500, "n_bc": 400,
            "n_train_supervised": 200, "n_mesh_nodes": 500,
        },
        "models": [
            {"name": "VanillaPINN", "type": "vanilla_pinn",
             "network": {"hidden": [64, 64, 64], "activation": "tanh"},
             "training": {"epochs": 2000, "lr": 1e-3}},
            {"name": "FNO-2D", "type": "fno2d",
             "network": {"width": 16, "modes": 8, "layers": 4},
             "training": {"epochs": 500, "lr": 1e-3}},
            {"name": "MeshGraphNet", "type": "meshgraphnet",
             "network": {"hidden_dim": 64, "n_message_passing": 4},
             "training": {"epochs": 500, "lr": 1e-3}},
        ],
        "output": {"dir": "outputs/poisson", "prefix": "benchmark",
                   "save_figures": True, "dpi": 150, "dark_theme": True},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Path to YAML or JSON config file")
    parser.add_argument("--problem", default="kovasznay_ns",
                        choices=list(BUILTIN_CONFIGS),
                        help="Built-in problem (ignored if --config is given)")
    args = parser.parse_args()

    if args.config:
        ext = os.path.splitext(args.config)[1].lower()
        if ext in (".yaml", ".yml"):
            arena = Arena.from_yaml(args.config)
        else:
            arena = Arena.from_json(args.config)
    else:
        cfg_dict = BUILTIN_CONFIGS[args.problem]
        arena = Arena(ArenaConfig.from_dict(cfg_dict))

    arena.run()


if __name__ == "__main__":
    main()
