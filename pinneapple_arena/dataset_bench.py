"""dataset_bench.py -- High-level benchmark API using pinneapple_data datasets.

Every dataset registered in ``pinneapple_data`` has a recommended preset here
(input/output fields, model list, training config, mode).  The public surface
is intentionally minimal:

  * :data:`DATASET_PRESETS`        -- full preset table, keyed by dataset_id
  * :func:`benchmark_dataset`      -- one-liner: load preset, build Arena, run
  * :func:`list_benchmarks`        -- show available presets (with category filter)
  * :func:`get_benchmark_preset`   -- retrieve a preset dict for inspection

Usage
-----
::

    from pinneapple_arena.dataset_bench import benchmark_dataset

    # Simplest usage -- all defaults from preset
    benchmark_dataset("burgers_1d", output_dir="outputs/burgers/")

    # Override epochs and model list
    benchmark_dataset(
        "kovasznay_ns",
        models=["VanillaPINN", "SIREN"],
        epochs=3000,
        mode="pinn_data",
        output_dir="outputs/kovasznay/",
    )

    # Regression dataset
    benchmark_dataset("airfoil_noise", output_dir="outputs/airfoil/")

    # List everything available
    from pinneapple_arena.dataset_bench import list_benchmarks
    list_benchmarks()
    list_benchmarks(category="pde")
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from .dataset_problems import DatasetProblem


# -----------------------------------------------------------------------
# Preset table
# -----------------------------------------------------------------------

#: Dataset presets: dataset_id -> configuration dict.
#:
#: Each entry has:
#:   category      : "pde" | "timeseries" | "geometry" | "regression" | "inverse"
#:   input_fields  : list of field names -> model inputs
#:   output_fields : list of field names -> model outputs
#:   mode          : "supervised" | "pinn_data" | "timeseries"
#:   models        : recommended model types for this problem class
#:   epochs        : recommended training epochs
#:   n_train       : recommended collocation / training point count
#:   n_bc          : recommended BC point count
#:   grid_n        : grid resolution for evaluation / visualisation
#:   dataset_params: kwargs forwarded to load_dataset()
#:   description   : human-readable summary

DATASET_PRESETS: Dict[str, Dict[str, Any]] = {

    # -- PDE benchmarks -------------------------------------------------------

    "burgers_1d": {
        "category": "pde",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN", "ModifiedMLP"],
        "epochs": 4000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 50,
        "dataset_params": {"Nx": 256, "Nt": 101},
        "description": "Viscous Burgers equation, 1-D+time, nu=0.01/pi",
    },
    "heat_1d": {
        "category": "pde",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 2000,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "description": "1-D heat/diffusion equation u_t = k*u_xx",
    },
    "wave_1d": {
        "category": "pde",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN"],
        "epochs": 4000,
        "n_train": 2500,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "description": "1-D wave equation u_tt = c^2*u_xx",
    },
    "poisson_2d": {
        "category": "pde",
        "input_fields": ["x", "y"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN", "FourierPINN"],
        "epochs": 4000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 40,
        "dataset_params": {},
        "description": "2-D Poisson equation Lap(u) = f on [0,1]^2",
    },
    "helmholtz_2d": {
        "category": "pde",
        "input_fields": ["x", "y"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN", "FourierPINN"],
        "epochs": 5000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 40,
        "dataset_params": {"k": 1.0, "a1": 1.0, "a2": 1.0},
        "description": "2-D Helmholtz equation u_xx+u_yy+k^2*u = f",
    },
    "allen_cahn_1d": {
        "category": "pde",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN", "ModifiedMLP"],
        "epochs": 5000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 40,
        "dataset_params": {"eps": 0.01},
        "description": "1-D Allen-Cahn phase-field equation (stiff, eps=0.01)",
    },
    "kovasznay_ns": {
        "category": "pde",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v", "p"],
        "mode": "pinn_data",
        "models": ["VanillaPINN", "SIREN"],
        "epochs": 5000,
        "n_train": 3000,
        "n_bc": 500,
        "grid_n": 40,
        "dataset_params": {"re": 40.0},
        "description": "Kovasznay 2-D Navier-Stokes steady flow (Re=40)",
    },
    "navier_stokes_2d": {
        "category": "pde",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v"],
        "mode": "supervised",
        "models": ["VanillaPINN", "FNO2d", "MeshGraphNet"],
        "epochs": 2000,
        "n_train": 4000,
        "n_bc": 400,
        "grid_n": 50,
        "dataset_params": {"n_samples": 1000},
        "description": "2-D incompressible Navier-Stokes velocity fields",
    },
    "diffusion_reaction": {
        "category": "pde",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "supervised",
        "models": ["VanillaPINN", "SIREN", "ModifiedMLP"],
        "epochs": 4000,
        "n_train": 2500,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "description": "1-D reaction-diffusion equation",
    },

    # -- Timeseries / ODE benchmarks ------------------------------------------

    "lorenz63": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["x", "y", "z"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 4000,
        "n_train": 3000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
        "description": "Lorenz-63 chaotic attractor (3-state ODE)",
    },
    "spring_mass": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["x", "v"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 1000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Spring-mass ODE: x'' + w^2*x = 0",
    },
    "van_der_pol": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["x", "y"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP", "SIREN"],
        "epochs": 4000,
        "n_train": 2000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {"mu": 1.0},
        "description": "Van der Pol oscillator (nonlinear ODE, mu=1)",
    },
    "duffing_oscillator": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["x", "v"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 1000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Duffing oscillator -- nonlinear spring with cubic term",
    },
    "double_pendulum": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["theta1", "theta2", "omega1", "omega2"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 5000,
        "n_train": 2000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Double pendulum -- 4-state chaotic Hamiltonian system",
    },
    "lotka_volterra": {
        "category": "timeseries",
        "input_fields": ["t"],
        "output_fields": ["x", "y"],
        "mode": "timeseries",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 1000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Lotka-Volterra predator-prey ODE",
    },

    # -- Geometry / CFD benchmarks --------------------------------------------

    "naca0012": {
        "category": "geometry",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v", "p"],
        "mode": "supervised",
        "models": ["VanillaPINN", "MeshGraphNet"],
        "epochs": 3000,
        "n_train": 4000,
        "n_bc": 500,
        "grid_n": 40,
        "dataset_params": {},
        "description": "NACA-0012 airfoil CFD -- velocity and pressure fields",
    },
    "cylinder_2d": {
        "category": "geometry",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v", "p"],
        "mode": "supervised",
        "models": ["VanillaPINN", "MeshGraphNet", "FNO2d"],
        "epochs": 3000,
        "n_train": 4000,
        "n_bc": 500,
        "grid_n": 40,
        "dataset_params": {},
        "description": "2-D flow past a cylinder -- CFD velocity and pressure",
    },
    "channel_flow": {
        "category": "geometry",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v"],
        "mode": "supervised",
        "models": ["VanillaPINN", "FNO2d"],
        "epochs": 2000,
        "n_train": 3000,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "description": "2-D plane channel (Poiseuille) flow",
    },
    "lid_driven_cavity": {
        "category": "geometry",
        "input_fields": ["x", "y"],
        "output_fields": ["u", "v", "p"],
        "mode": "supervised",
        "models": ["VanillaPINN", "SIREN", "MeshGraphNet"],
        "epochs": 4000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 40,
        "dataset_params": {"re": 100.0},
        "description": "Lid-driven cavity 2-D Navier-Stokes (Re=100)",
    },
    "turbulent_channel": {
        "category": "geometry",
        "input_fields": ["y"],
        "output_fields": ["u_mean"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 2000,
        "n_train": 1000,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Turbulent channel flow mean velocity profile",
    },
    "darcy_flow": {
        "category": "geometry",
        "input_fields": ["x", "y"],
        "output_fields": ["p"],
        "mode": "supervised",
        "models": ["VanillaPINN", "FNO2d"],
        "epochs": 3000,
        "n_train": 2000,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "description": "2-D Darcy flow pressure field in porous media",
    },

    # -- Regression / real-world benchmarks -----------------------------------

    "airfoil_noise": {
        "category": "regression",
        "input_fields": ["frequency", "angle_attack", "chord_length",
                         "velocity", "displacement"],
        "output_fields": ["sound_pressure_level"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 1200,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "UCI Airfoil Self-Noise regression (5 features -> SPL)",
    },
    "concrete_strength": {
        "category": "regression",
        "input_fields": ["cement", "blast_furnace_slag", "fly_ash", "water",
                         "superplasticizer", "coarse_aggregate",
                         "fine_aggregate", "age"],
        "output_fields": ["compressive_strength"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 3000,
        "n_train": 800,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Concrete compressive strength (8 mix features -> MPa)",
    },
    "energy_efficiency": {
        "category": "regression",
        "input_fields": ["relative_compactness", "surface_area", "wall_area",
                         "roof_area", "overall_height", "orientation",
                         "glazing_area", "glazing_area_distribution"],
        "output_fields": ["heating_load"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 2000,
        "n_train": 600,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Building energy efficiency -- heating load prediction",
    },
    "wine_quality": {
        "category": "regression",
        "input_fields": ["fixed_acidity", "volatile_acidity", "citric_acid",
                         "residual_sugar", "chlorides", "free_sulfur_dioxide",
                         "total_sulfur_dioxide", "density", "pH",
                         "sulphates", "alcohol"],
        "output_fields": ["quality"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 2000,
        "n_train": 1200,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Wine quality regression (11 physicochemical features)",
    },
    "boston_housing": {
        "category": "regression",
        "input_fields": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                         "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
        "output_fields": ["MEDV"],
        "mode": "supervised",
        "models": ["VanillaPINN", "ModifiedMLP"],
        "epochs": 2000,
        "n_train": 400,
        "n_bc": 0,
        "grid_n": 30,
        "dataset_params": {},
        "description": "Boston housing price prediction (13 features)",
    },

    # -- Inverse problem benchmarks -------------------------------------------

    "burgers_1d_inverse": {
        "category": "inverse",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN"],
        "epochs": 5000,
        "n_train": 3000,
        "n_bc": 400,
        "grid_n": 40,
        "dataset_params": {"Nx": 256, "Nt": 101},
        "inverse_params": ["nu"],
        "description": "Burgers 1-D inverse: infer viscosity nu from data",
    },
    "heat_1d_inverse": {
        "category": "inverse",
        "input_fields": ["x", "t"],
        "output_fields": ["u"],
        "mode": "pinn_data",
        "models": ["VanillaPINN"],
        "epochs": 5000,
        "n_train": 2000,
        "n_bc": 300,
        "grid_n": 40,
        "dataset_params": {},
        "inverse_params": ["k"],
        "description": "Heat equation inverse: infer thermal diffusivity k",
    },
}

# Convenience alias: category -> list of dataset_ids
_CATEGORY_INDEX: Dict[str, List[str]] = {}
for _did, _p in DATASET_PRESETS.items():
    _CATEGORY_INDEX.setdefault(_p["category"], []).append(_did)


# -----------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------

def list_benchmarks(category: Optional[str] = None, verbose: bool = True) -> List[str]:
    """Print and return available benchmark preset IDs.

    Parameters
    ----------
    category : filter by ``"pde"``, ``"timeseries"``, ``"geometry"``,
               ``"regression"``, or ``"inverse"``.  ``None`` shows all.
    verbose  : if True, print a formatted table.

    Returns
    -------
    List of dataset IDs matching the filter.
    """
    if category is not None and category not in _CATEGORY_INDEX:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Available: {sorted(_CATEGORY_INDEX.keys())}"
        )

    cats = [category] if category else sorted(_CATEGORY_INDEX.keys())
    ids: List[str] = []

    if verbose:
        print("\n" + "-" * 70)
        print("  PINNeAPPle Arena -- Dataset Benchmarks")
        print("-" * 70)

    for cat in cats:
        cat_ids = _CATEGORY_INDEX.get(cat, [])
        ids.extend(cat_ids)
        if verbose:
            print(f"\n  [{cat.upper()}]")
            for did in cat_ids:
                desc = DATASET_PRESETS[did]["description"]
                print(f"    {did:<30}  {desc}")

    if verbose:
        print("-" * 70)
        print(f"  Total: {len(ids)} benchmarks\n")

    return ids


def get_benchmark_preset(dataset_id: str) -> Dict[str, Any]:
    """Return the preset configuration dict for a dataset_id.

    Raises
    ------
    KeyError if the dataset_id is not in DATASET_PRESETS.
    """
    if dataset_id not in DATASET_PRESETS:
        raise KeyError(
            f"No preset for '{dataset_id}'. "
            f"Call list_benchmarks() to see available IDs."
        )
    return dict(DATASET_PRESETS[dataset_id])


# -----------------------------------------------------------------------
# One-liner benchmark runner
# -----------------------------------------------------------------------

def benchmark_dataset(
    dataset_id: str,
    *,
    models: Optional[List[str]] = None,
    epochs: Optional[int] = None,
    mode: Optional[str] = None,
    n_train: Optional[int] = None,
    n_bc: Optional[int] = None,
    grid_n: Optional[int] = None,
    output_dir: str = "outputs/dataset_bench/",
    prefix: Optional[str] = None,
    hidden: Optional[List[int]] = None,
    lr: float = 1e-3,
    dark_theme: bool = True,
    save_figures: bool = True,
    show: bool = False,
    uq: bool = False,
    inverse: bool = False,
    name: Optional[str] = None,
    description: str = "",
    **dataset_params,
) -> "DatasetProblem":
    """Run a full Arena benchmark on a pinneapple_data dataset.

    The preset for ``dataset_id`` supplies all defaults; any keyword argument
    overrides the preset value.

    Parameters
    ----------
    dataset_id   : dataset registered in ``pinneapple_data``
                   (see :func:`list_benchmarks` for all IDs).
    models       : list of model type strings.  Defaults from preset.
    epochs       : training epochs per model.  Defaults from preset.
    mode         : ``"supervised"`` | ``"pinn_data"`` | ``"timeseries"``.
    n_train      : number of training / collocation points.
    n_bc         : number of boundary-condition points.
    grid_n       : grid size for evaluation grid.
    output_dir   : root output directory.
    prefix       : filename prefix (defaults to ``dataset_id``).
    hidden       : hidden layer sizes for MLP-based models.
    lr           : learning rate.
    dark_theme   : use dark matplotlib theme.
    save_figures : save plots to ``output_dir``.
    show         : call ``plt.show()`` after each figure.
    uq           : enable Monte-Carlo Dropout uncertainty quantification.
    inverse      : enable inverse-problem mode (preset must have
                   ``inverse_params``).
    name         : Arena problem name (defaults to ``dataset_id``).
    description  : human-readable description.
    **dataset_params : extra kwargs forwarded to ``load_dataset()``.

    Returns
    -------
    DatasetProblem  (already run -- results are in ``output_dir``).
    """
    # merge preset + overrides
    if dataset_id in DATASET_PRESETS:
        preset = dict(DATASET_PRESETS[dataset_id])
    else:
        warnings.warn(
            f"[benchmark_dataset] No preset found for '{dataset_id}'. "
            "Using defaults -- specify input_fields and output_fields via "
            "dataset_params if needed."
        )
        preset = {
            "category": "unknown",
            "input_fields": dataset_params.pop("input_fields", []),
            "output_fields": dataset_params.pop("output_fields", []),
            "mode": "supervised",
            "models": ["VanillaPINN"],
            "epochs": 2000,
            "n_train": 1000,
            "n_bc": 200,
            "grid_n": 30,
            "dataset_params": {},
            "description": "",
        }

    _models    = models  or preset["models"]
    _epochs    = epochs  or preset["epochs"]
    _mode      = mode    or preset["mode"]
    _n_train   = n_train or preset["n_train"]
    _n_bc      = n_bc    if n_bc is not None else preset["n_bc"]
    _grid_n    = grid_n  or preset["grid_n"]
    _desc      = description or preset.get("description", "")
    _prefix    = prefix  or dataset_id

    # dataset_params in preset take lower priority than caller kwargs
    _ds_params = {**preset.get("dataset_params", {}), **dataset_params}

    _input_fields  = preset["input_fields"]
    _output_fields = preset["output_fields"]

    # build problem
    prob = DatasetProblem.from_dataset(
        dataset_id,
        input_fields=_input_fields,
        output_fields=_output_fields,
        mode=_mode,
        name=name or dataset_id,
        description=_desc,
        register=True,
        **_ds_params,
    )

    # build Arena config
    _hidden = hidden or [128, 128, 128, 128]

    def _model_cfg(model_name: str) -> dict:
        return {
            "name": model_name,
            "type": _model_type(model_name),
            "network": {"hidden": _hidden},
            "training": {"epochs": _epochs, "lr": lr},
        }

    arena_dict: Dict[str, Any] = {
        "problem": {"name": prob.name},
        "models": [_model_cfg(m) for m in _models],
        "output": {"dir": f"{output_dir}/{_prefix}/", "prefix": _prefix},
    }

    if uq:
        arena_dict["uq"] = {"enabled": True, "method": "mc_dropout", "n_samples": 30}

    if inverse:
        inv_params = preset.get("inverse_params", [])
        if inv_params:
            arena_dict["inverse"] = {
                "enabled": True,
                "params": inv_params,
                "n_iters": min(_epochs, 3000),
            }
        else:
            warnings.warn(
                f"[benchmark_dataset] inverse=True but no 'inverse_params' "
                f"in preset for '{dataset_id}'. Skipping inverse mode."
            )

    # run
    from .arena import Arena
    from .config import ArenaConfig

    cfg = ArenaConfig.from_dict(arena_dict)
    arena = Arena(cfg)

    print(f"\n{'='*62}")
    print(f"  Dataset Benchmark: {dataset_id}")
    print(f"  Mode      : {_mode}")
    print(f"  Models    : {', '.join(_models)}")
    print(f"  Epochs    : {_epochs}")
    print(f"  n_train   : {_n_train}   n_bc: {_n_bc}   grid_n: {_grid_n}")
    print(f"{'='*62}\n")

    arena.run(
        n_col=_n_train,
        n_bc=_n_bc,
        grid_n=_grid_n,
        dark_theme=dark_theme,
        save_figures=save_figures,
        show=show,
    )

    return prob


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

_MODEL_TYPE_MAP = {
    "VanillaPINN":   "vanilla_pinn",
    "SIREN":         "siren",
    "ModifiedMLP":   "modified_mlp",
    "FourierPINN":   "fourier_pinn",
    "FNO2d":         "fno2d",
    "MeshGraphNet":  "meshgraphnet",
    "DeepONet":      "deeponet",
    "AFNO":          "afno",
}


def _model_type(name: str) -> str:
    """Map a friendly model name to an Arena type string."""
    return _MODEL_TYPE_MAP.get(name, name.lower())


__all__ = [
    "DATASET_PRESETS",
    "list_benchmarks",
    "get_benchmark_preset",
    "benchmark_dataset",
]
