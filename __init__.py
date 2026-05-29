"""
pinneapple — Physics-Informed Neural Networks for Engineering and Research.

A unified library for building, training, and deploying physics-informed
surrogate models and digital twins at research and industrial scale.

Quick start
-----------
>>> import pinneapple as pp

>>> # List all available problem presets
>>> pp.list_presets()

>>> # Load a problem
>>> spec = pp.get_preset("burgers_1d", nu=0.01)

>>> # Check available models
>>> pp.list_models()

>>> # Train a PINN (see examples/ for full workflows)
>>> model = pp.build_model("VanillaPINN", in_dim=2, out_dim=1)

>>> # Create a digital twin
>>> dt = pp.build_digital_twin(model, field_names=["u"])

>>> # Full-pipeline quickstart (generates geometry, trains, visualizes)
>>> pp.quickstart("burgers_1d")

Submodules
----------
Core:
- pinneapple_environment   problem presets, PDE specs, BCs
- pinneapple_models        PINN, DeepONet, FNO, GNO, GNN, autoencoders, ...
- pinneapple_train         Trainer, metrics, AMP, parallelization, sweeps
- pinneapple_pinn          physics loss compiler (PINNFactory)
- pinneapple_solvers       FDM, FEM, FVM, SPH, OpenFOAM, FEniCS bridges
- pinneapple_data          collocation samplers, active learning, dataset builders
- pinneapple_geom          geometry generation, SDF, mesh, domains
- pinneapple_inference     grid inference, error maps, visualization
- pinneapple_digital_twin  digital twin runtime, sensor streams, anomaly detection
- pinneapple_arena         benchmark runner, YAML experiments, end-to-end pipeline

Advanced:
- pinneapple_uq            uncertainty quantification (MC Dropout, ensemble, conformal)
- pinneapple_transfer      transfer learning and parametric fine-tuning
- pinneapple_meta          meta-learning: MAML and Reptile for PDE families
- pinneapple_validate      physical consistency validation (conservation, BCs, symmetry)
- pinneapple_serve         REST API inference server (FastAPI)
- pinneapple_export        model export to ONNX and TorchScript

Examples
--------
See examples/pinneapple_arena/ for ready-to-run scripts:
  01_quickstart_native.py           — Basic benchmark run
  03_pinn_burgers_full_pipeline.py  — Full PINN training with GPU + viz
  04_digital_twin_flow.py           — Live digital twin with anomaly detection
  05_surrogate_deeponet_multifield.py — DeepONet operator learning
  06_engineering_presets_showcase.py — Aerospace, automotive, datacenter presets
  07_xtfc_1d_ode.py                 — XtFC with TFC library
  08_hyperparameter_sweep_parallel.py — Parallel HP sweep
  09_full_pipeline_yaml.py          — End-to-end YAML pipeline
  10_datacenter_digital_twin.py     — Industrial digital twin
"""

from __future__ import annotations

__version__ = "0.4.0"
__author__  = "pinneapple contributors"

# ---------------------------------------------------------------------------
# Convenience top-level imports
# ---------------------------------------------------------------------------

# Problem presets
try:
    from pinneapple_physics.pde_environment import get_preset, list_presets, register_preset
except Exception:  # pragma: no cover
    pass

# Model builder
try:
    from pinneapple_neural.architectures import ModelRegistry

    def list_models():
        """Return sorted list of all registered model names."""
        return ModelRegistry.list()

    def build_model(name: str, **kwargs):
        """Build a model by name from the model registry."""
        return ModelRegistry.build(name, **kwargs)

except Exception:  # pragma: no cover
    def list_models():
        return []
    def build_model(name, **kwargs):
        raise ImportError("pinneapple_models not available")

# Digital twin
try:
    from pinneapple_systems.digital_twin import build_digital_twin, DigitalTwin, DigitalTwinConfig
except Exception:  # pragma: no cover
    pass

# Inference
try:
    from pinneapple_neural.predictor import infer_on_grid_1d, infer_on_grid_2d
except Exception:  # pragma: no cover
    pass

# Training
try:
    from pinneapple_neural.trainer import (
        Trainer, TrainConfig,
        best_device, count_gpus, gpu_info,
        maybe_compile, batched_inference,
        run_parallel_sweep, SweepConfig,
        ThroughputMonitor,
        WeightScheduler, WeightSchedulerConfig,
        SelfAdaptiveWeights, GradNormBalancer, LossRatioBalancer,
    )
except Exception:  # pragma: no cover
    pass

# Data + Collocation + Active Learning
try:
    from pinneapple_data import (
        CollocationSampler, CollocationConfig,
        ActiveLearningConfig, ResidualBasedAL, AdaptiveCollocationTrainer,
    )
except Exception:  # pragma: no cover
    pass

# Geometry
try:
    from pinneapple_design.geometry import (
        circle, rectangle, ellipse, annulus,
        ChannelWithObstacleDomain2D, ChannelDomain2D, LidDrivenCavityDomain2D,
        PhysicsDomain2D,
    )
except Exception:  # pragma: no cover
    pass

# UQ
try:
    from pinneapple_analysis.uncertainty import MCDropoutWrapper, EnsembleUQ, ConformalPredictor, uq_predict
except Exception:  # pragma: no cover
    pass

# Validation
try:
    from pinneapple_analysis.validation import PhysicsValidator, validate_model
except Exception:  # pragma: no cover
    pass

# Export
try:
    from pinneapple_tools.model_export import export_torchscript, export_onnx
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Quickstart helper
# ---------------------------------------------------------------------------

def quickstart(problem_id: str = "burgers_1d", **problem_kwargs):
    """
    Interactive quickstart that summarises a problem preset and its fields.

    Parameters
    ----------
    problem_id : str
        Name of the problem preset (e.g. "burgers_1d", "datacenter_airflow_2d").
    **problem_kwargs
        Extra parameters passed to the preset factory.

    Example
    -------
    >>> import pinneapple as pp
    >>> pp.quickstart("cpu_heatsink_thermal", q_cpu=200.0)
    """
    try:
        spec = get_preset(problem_id, **problem_kwargs)
    except Exception as e:
        print(f"[pinneapple] Could not load preset '{problem_id}': {e}")
        print(f"  Available presets: {list_presets()}")
        return

    border = "=" * 60
    print(border)
    print(f"  pinneapple — Quickstart: {spec.problem_id}")
    print(border)
    print(f"  PDE kind    : {spec.pde.kind}")
    print(f"  Fields      : {spec.fields}")
    print(f"  Coordinates : {spec.coord_names}")
    print(f"  Domain      : {spec.domain_bounds}")
    print(f"  Conditions  : {list(spec.conditions)}")
    print(f"  Solver      : {spec.solver_spec.get('name', 'N/A')}")
    if spec.meta.get("description"):
        print(f"  Description : {spec.meta['description']}")
    if spec.meta.get("digital_twin_fields"):
        print(f"  DT fields   : {spec.meta['digital_twin_fields']}")
    print(border)
    print()
    print("  Next steps:")
    print(f"  1. from pinneapple_physics.pde_environment import get_preset")
    print(f"     spec = get_preset('{problem_id}')")
    print(f"  2. from pinneapple_data import CollocationSampler")
    print(f"     sampler = CollocationSampler.from_problem_spec(spec)")
    print(f"     batch = sampler.sample(n_col=8000, n_bc=1000)")
    print(f"  3. model = pp.build_model('VanillaPINN', in_dim={len(spec.coord_names)}, out_dim={len(spec.fields)})")
    print(f"  4. See examples/pinneapple_arena/03_pinn_burgers_full_pipeline.py")
    print(border)
    return spec


def info():
    """Print library version, device info, and available presets/models."""
    print(f"pinneapple v{__version__}")
    try:
        d = best_device()
        print(f"  Best device  : {d}")
        print(f"  GPUs         : {count_gpus()}")
        for g in gpu_info():
            print(f"    [{g['index']}] {g['name']}  {g['total_memory_GB']:.1f} GB  "
                  f"CC={g['compute_capability']}")
    except Exception:
        pass

    try:
        presets = list_presets()
        print(f"  Problem presets  : {len(presets)}")
    except Exception:
        pass

    try:
        models = list_models()
        print(f"  Registered models: {len(models)}")
    except Exception:
        pass

    # New modules status
    new_modules = {
        "pinneapple_uq":       "Uncertainty quantification",
        "pinneapple_transfer": "Transfer learning",
        "pinneapple_meta":     "Meta-learning (MAML/Reptile)",
        "pinneapple_validate": "Physical validation",
        "pinneapple_serve":    "REST inference server",
        "pinneapple_export":   "Model export (ONNX/TorchScript)",
        "pinneapple_quantum":  "Hybrid classical–quantum ML (PQM)",
    }
    import importlib
    print()
    print("  Advanced modules:")
    for mod, desc in new_modules.items():
        try:
            importlib.import_module(mod)
            status = "OK"
        except ImportError:
            status = "optional deps missing"
        except Exception as e:
            status = f"error: {e}"
        print(f"    {mod:<26} [{status}]  — {desc}")


# ---------------------------------------------------------------------------
# Lazy submodule access (pinneapple.train, pinneapple.models, etc.)
# ---------------------------------------------------------------------------

import importlib as _importlib
import sys as _sys

_SUBMODULES = {
    # core
    "env":       "pinneapple_environment",
    "models":    "pinneapple_models",
    "train":     "pinneapple_train",
    "solvers":   "pinneapple_solvers",
    "data":      "pinneapple_data",
    "geom":      "pinneapple_geom",
    "inference": "pinneapple_inference",
    "pinn":      "pinneapple_pinn",
    "dt":        "pinneapple_digital_twin",
    "arena":     "pinneapple_arena",
    # advanced
    "uq":        "pinneapple_uq",
    "transfer":  "pinneapple_transfer",
    "meta":      "pinneapple_meta",
    "validate":  "pinneapple_validate",
    "serve":     "pinneapple_serve",
    "export":    "pinneapple_export",
    # quantum
    "quantum":   "pinneapple_quantum",
}


def __getattr__(name: str):
    """Enable lazy access: ``pinneapple.train`` → pinneapple_train."""
    if name in _SUBMODULES:
        mod = _importlib.import_module(_SUBMODULES[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'pinneapple' has no attribute '{name}'")


__all__ = [
    "__version__",
    # Problem
    "get_preset", "list_presets", "register_preset",
    # Models
    "list_models", "build_model",
    # Training
    "Trainer", "TrainConfig", "best_device", "count_gpus", "gpu_info",
    "maybe_compile", "batched_inference", "run_parallel_sweep", "SweepConfig",
    "ThroughputMonitor",
    "WeightScheduler", "WeightSchedulerConfig",
    "SelfAdaptiveWeights", "GradNormBalancer", "LossRatioBalancer",
    # Digital twin
    "DigitalTwin", "DigitalTwinConfig", "build_digital_twin",
    # Inference
    "infer_on_grid_1d", "infer_on_grid_2d",
    # Data + Active Learning
    "CollocationSampler", "CollocationConfig",
    "ActiveLearningConfig", "ResidualBasedAL", "AdaptiveCollocationTrainer",
    # Geometry
    "circle", "rectangle", "ellipse", "annulus",
    "ChannelDomain2D", "ChannelWithObstacleDomain2D",
    "LidDrivenCavityDomain2D", "PhysicsDomain2D",
    # UQ
    "MCDropoutWrapper", "EnsembleUQ", "ConformalPredictor", "uq_predict",
    # Validation
    "PhysicsValidator", "validate_model",
    # Export
    "export_torchscript", "export_onnx",
    # Helpers
    "quickstart", "info",
    # Lazy submodule aliases (core)
    "env", "models", "train", "solvers", "data", "geom",
    "inference", "pinn", "dt", "arena",
    # Lazy submodule aliases (advanced)
    "uq", "transfer", "meta", "validate", "serve", "export",
    # Quantum
    "quantum",
]
