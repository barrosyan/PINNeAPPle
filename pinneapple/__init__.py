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
- pinneapple_environment   problem presets, PDE specs, BCs, RANS turbulence presets
- pinneapple_models        PINN, DeepONet, FNO, GNO, GNN, autoencoders, SIREN,
                          ModifiedMLP, HashGridMLP, MeshGraphNet, AFNO
- pinneapple_train         Trainer, metrics, AMP, parallelization, sweeps,
                          TimeMarchingTrainer, DDPPINNTrainer, CausalPINNTrainer
- pinneapple_pinn          physics loss compiler (PINNFactory), DoMINO decomposition
- pinneapple_solvers       FDM, FEM, FVM, SPH, OpenFOAM, FEniCS bridges,
                          CADToCFDPipeline, NSFlowSolver, CFDMesh
- pinneapple_data          collocation samplers, active learning, dataset builders
- pinneapple_geom          geometry generation, SDF, mesh, CSG domains
- pinneapple_inference     grid inference, error maps, streamlines, isosurfaces
- pinneapple_digital_twin  digital twin runtime, sensor streams, anomaly detection
- pinneapple_arena         benchmark runner, YAML experiments, end-to-end pipeline

Advanced:
- pinneapple_symbolic      symbolic PDE compiler (SymPy → autograd), HardBC, PeriodicBC
- pinneapple_uq            uncertainty quantification (MC Dropout, ensemble, conformal)
- pinneapple_transfer      transfer learning and parametric fine-tuning
- pinneapple_meta          meta-learning: MAML and Reptile for PDE families
- pinneapple_validate      physical consistency validation (conservation, BCs, symmetry)
- pinneapple_serve         REST API inference server (FastAPI)
- pinneapple_export        model export to ONNX and TorchScript
- pinneapple_backend       multi-backend support: PyTorch + JAX (vmap/jit)
- pinneapple_dynamics      differentiable dynamics: rigid body, MPM, SPH particles
- pinneapple_worldmodel    world foundation model integration (NVIDIA Cosmos adapter)

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
  12_physics_benchmark_suite.py     — Multi-architecture PINN Arena benchmark
  13_transfer_meta_benchmark.py     — Transfer & meta-learning benchmark

New feature examples:
  examples/pinneapple_pinn/03_symbolic_pde_hard_bc.py   — Symbolic PDE + HardBC
  examples/pinneapple_models/60_new_architectures_demo.py — SIREN/ModMLP/AFNO/etc.
  examples/pinneapple_pinn/06_domino_time_marching_demo.py — DoMINO + time-marching
  examples/pinneapple_geom/07_csg_domain_demo.py         — CSG L-shape/annulus domains
  examples/pinneapple_solvers/11_cad_cfd_pipeline_demo.py — CAD→mesh→NS→PINN pipeline
"""

from __future__ import annotations

__version__ = "0.5.0"
__author__  = "pinneapple contributors"

# ---------------------------------------------------------------------------
# Convenience top-level imports
# ---------------------------------------------------------------------------

# Problem presets
try:
    from pinneapple_physics.pde_environment import get_preset, list_presets, register_preset
except Exception:  # pragma: no cover
    pass

# PDE family knowledge base (lives in pinneapple_environment.capabilities)
try:
    from pinneapple_physics.pde_environment import (
        list_pde_families,
        get_pde_family,
        identify_pde,
        suggest_problem_spec,
    )
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
        TwoPhaseTrainer, TwoPhaseConfig, TwoPhaseHistory,
        UnnormModel,
        best_device, count_gpus, gpu_info,
        maybe_compile, batched_inference,
        run_parallel_sweep, SweepConfig,
        ThroughputMonitor,
        WeightScheduler, WeightSchedulerConfig,
        SelfAdaptiveWeights, GradNormBalancer, LossRatioBalancer, NTKWeightBalancer,
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
    from pinneapple_tools.model_export import export_torchscript, export_onnx, export_csv, export_npz
except Exception:  # pragma: no cover
    pass

# Design optimization
try:
    from pinneapple_design.design_optimizer import (
        DesignOptLoop, DesignOptConfig, DesignOptResult,
        PhysicsSurrogate, SurrogateConfig,
        DragObjective, ThermalEfficiencyObjective, StructuralObjective,
        WeightMinimizationObjective, CompositeObjective,
        BoxConstraint, MassConservationConstraint, ConstraintSet,
        DesignOptimizerConfig,
        ParetoFront, compute_pareto_front,
        PINNRefinement,
    )
except Exception:  # pragma: no cover
    pass

# Inverse problems
try:
    from pinneapple_analysis.inverse_problems import (
        # Noise models
        GaussianMisfit, HuberMisfit, CauchyMisfit, StudentTMisfit, HeteroscedasticMisfit,
        # Regularization
        TikhonovRegularizer, SparsityRegularizer, TotalVariationRegularizer,
        CompositeRegularizer, LCurveSelector,
        # Observation operators
        PointObsOperator, LinearObsOperator, IntegralObsOperator, ComposedObsOperator,
        # Sensitivity
        LocalSensitivity, IdentifiabilityAnalyzer, GlobalSensitivity,
        # EKI
        EKIConfig, EnsembleKalmanInversion, IteratedEKI,
        # Solver
        InverseSolverConfig, InverseProblemSolver,
    )
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# New features: Symbolic PDE compiler + BC enforcement (Features 1, 4, 5)
# ---------------------------------------------------------------------------
try:
    from pinneapple_physics.symbolic_pde import (
        SymbolicPDE, pde_from_sympy, auto_residual,
        HardBC, PeriodicBC, DirichletBC, NeumannBC,
    )
except Exception:  # pragma: no cover
    pass

# New architectures (Features 2, 3, 6, 7, 8)
try:
    from pinneapple_neural.architectures import (
        SIREN, SineLayer,
        ModifiedMLP, FourierFeatureEmbedding,
        HashGridEncoding, HashGridMLP,
        MeshGraphNet,
        AFNO,
    )
except Exception:  # pragma: no cover
    pass

# DoMINO domain decomposition (Feature 9)
try:
    from pinneapple_physics.pinn_solver import (
        DoMINO, Subdomain, SubdomainPINN,
    )
except Exception:  # pragma: no cover
    pass

# Advanced training: time-marching, DDP, causal (Features 11, 13, 14)
try:
    from pinneapple_neural.trainer import (
        TimeMarchingTrainer,
        DDPTrainerConfig, DDPPINNTrainer,
        is_distributed, get_rank, get_world_size,
        CausalWeightScheduler, CausalPINNTrainer,
    )
except Exception:  # pragma: no cover
    pass

# RANS turbulence presets (Feature 10)
try:
    from pinneapple_physics.pde_environment import (
        KOmegaSSTResiduals, SpalartAllmarasResiduals, get_rans_preset,
    )
except Exception:  # pragma: no cover
    pass

# CSG geometry (Feature 12)
try:
    from pinneapple_design.geometry import (
        CSGRectangle, CSGCircle, CSGEllipse, CSGPolygon,
        CSGUnion, CSGIntersection, CSGDifference,
        lshape, csg_annulus, channel_with_hole, t_junction,
    )
except Exception:  # pragma: no cover
    pass

# Streamline / isosurface post-processing (Feature 17)
try:
    from pinneapple_neural.predictor import (
        FlowVisualizer, compute_streamlines, compute_isosurface,
        plot_streamlines_2d_model, plot_isosurface_3d, plot_volume_slice,
    )
except Exception:  # pragma: no cover
    pass

# Multi-backend JAX support (Feature 15)
try:
    from pinneapple_tools.compute_backends import (
        get_backend, set_backend, Backend,
        JAXBackend, jit_pinn, vmap_residual,
    )
except Exception:  # pragma: no cover
    pass

# Differentiable dynamics: rigid body, MPM, particles (Feature 18)
try:
    from pinneapple_simulation.particle_dynamics import (
        RigidBody, RigidBodySystem, RigidBodyState,
        MPMSimulator, MPMState,
        SPHParticles, ParticleSystem,
    )
except Exception:  # pragma: no cover
    pass

# World foundation model integration (Feature 19)
try:
    from pinneapple_worldmodel import (
        CosmosAdapter, WorldModelConfig,
        PhysicsVideoDataset, SimToRealAdapter,
        PhysicalScene, SceneObject,
    )
except Exception:  # pragma: no cover
    pass

# Adjoint shape optimization (Feature 16)
try:
    from pinneapple_design.design_optimizer import (
        ContinuousAdjointSolver, ShapeParametrization,
        DragAdjointObjective, naca_parametric,
    )
except Exception:  # pragma: no cover
    pass

# CAD → mesh → NS CFD pipeline (Feature 20)
try:
    from pinneapple_simulation.numerical_solvers import (
        CFDMesh, NSFlowSolver, CADToCFDPipeline,
    )
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------
try:
    from pinneapple_tools.benchmark_suite import (
        PINNArenaBenchmark, BenchmarkConfig, BenchmarkResult,
        BenchmarkTaskBase, ModelSpec, DEFAULT_MODELS,
    )
except Exception:  # pragma: no cover
    pass

# Transfer learning benchmark
try:
    from pinneapple_tools.benchmark_suite import (
        TransferBenchmarkPipeline, TransferBenchmarkConfig,
        TransferBenchmarkResult, TransferScenario,
    )
except Exception:  # pragma: no cover
    pass

# Meta-learning benchmark
try:
    from pinneapple_tools.benchmark_suite import (
        MetaBenchmarkPipeline, MetaBenchmarkConfig,
        MetaBenchmarkResult, MetaBenchmarkFamily,
    )
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
        "pinneapple_uq":         "Uncertainty quantification",
        "pinneapple_transfer":   "Transfer learning",
        "pinneapple_meta":       "Meta-learning (MAML/Reptile)",
        "pinneapple_validate":   "Physical validation",
        "pinneapple_serve":      "REST inference server",
        "pinneapple_export":     "Model export (ONNX/TorchScript)",
        "pinneapple_quantum":    "Hybrid classical–quantum ML (PQM)",
        # v0.5 new modules
        "pinneapple_symbolic":   "Symbolic PDE compiler + HardBC/PeriodicBC",
        "pinneapple_backend":    "Multi-backend (PyTorch + JAX)",
        "pinneapple_dynamics":   "Differentiable dynamics (rigid body, MPM, SPH)",
        "pinneapple_worldmodel": "World foundation model integration (Cosmos)",
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
    "env":        "pinneapple_environment",
    "inverse":    "pinneapple_inverse",
    "design_opt": "pinneapple_design_opt",
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
    # new features (v0.5)
    "symbolic":   "pinneapple_symbolic",
    "backend":    "pinneapple_backend",
    "dynamics":   "pinneapple_dynamics",
    "worldmodel": "pinneapple_worldmodel",
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
    # PDE knowledge base
    "list_pde_families", "get_pde_family", "identify_pde", "suggest_problem_spec",
    # Models
    "list_models", "build_model",
    # Training
    "Trainer", "TrainConfig", "best_device", "count_gpus", "gpu_info",
    "maybe_compile", "batched_inference", "run_parallel_sweep", "SweepConfig",
    "ThroughputMonitor",
    "WeightScheduler", "WeightSchedulerConfig",
    "SelfAdaptiveWeights", "GradNormBalancer", "LossRatioBalancer", "NTKWeightBalancer",
    "TwoPhaseTrainer", "TwoPhaseConfig", "TwoPhaseHistory", "UnnormModel",
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
    "export_torchscript", "export_onnx", "export_csv", "export_npz",
    # Inverse problems
    "GaussianMisfit", "HuberMisfit", "CauchyMisfit", "StudentTMisfit", "HeteroscedasticMisfit",
    "TikhonovRegularizer", "SparsityRegularizer", "TotalVariationRegularizer",
    "CompositeRegularizer", "LCurveSelector",
    "PointObsOperator", "LinearObsOperator", "IntegralObsOperator", "ComposedObsOperator",
    "LocalSensitivity", "IdentifiabilityAnalyzer", "GlobalSensitivity",
    "EKIConfig", "EnsembleKalmanInversion", "IteratedEKI",
    "InverseSolverConfig", "InverseProblemSolver",
    # Helpers
    "quickstart", "info",
    # Design optimization
    "DesignOptLoop", "DesignOptConfig", "DesignOptResult",
    "PhysicsSurrogate", "SurrogateConfig",
    "DragObjective", "ThermalEfficiencyObjective", "StructuralObjective",
    "WeightMinimizationObjective", "CompositeObjective",
    "BoxConstraint", "MassConservationConstraint", "ConstraintSet",
    "DesignOptimizerConfig", "ParetoFront", "compute_pareto_front",
    "PINNRefinement",
    # Benchmark suite
    "PINNArenaBenchmark", "BenchmarkConfig", "BenchmarkResult",
    "BenchmarkTaskBase", "ModelSpec", "DEFAULT_MODELS",
    # Transfer learning benchmark
    "TransferBenchmarkPipeline", "TransferBenchmarkConfig",
    "TransferBenchmarkResult", "TransferScenario",
    # Meta-learning benchmark
    "MetaBenchmarkPipeline", "MetaBenchmarkConfig",
    "MetaBenchmarkResult", "MetaBenchmarkFamily",
    # Lazy submodule aliases (core)
    "env", "models", "train", "solvers", "data", "geom",
    "inference", "pinn", "dt", "arena", "design_opt",
    # Lazy submodule aliases (advanced)
    "uq", "transfer", "meta", "validate", "serve", "export",
    # Quantum
    "quantum",
    # ---- v0.5 new features ----
    # Symbolic PDE compiler + BC (Features 1, 4, 5)
    "SymbolicPDE", "pde_from_sympy", "auto_residual",
    "HardBC", "PeriodicBC", "DirichletBC", "NeumannBC",
    # New architectures (Features 2, 3, 6, 7, 8)
    "SIREN", "SineLayer",
    "ModifiedMLP", "FourierFeatureEmbedding",
    "HashGridEncoding", "HashGridMLP",
    "MeshGraphNet",
    "AFNO",
    # Domain decomposition PINN (Feature 9)
    "DoMINO", "Subdomain", "SubdomainPINN",
    # Advanced training (Features 11, 13, 14)
    "TimeMarchingTrainer",
    "DDPTrainerConfig", "DDPPINNTrainer",
    "is_distributed", "get_rank", "get_world_size",
    "CausalWeightScheduler", "CausalPINNTrainer",
    # RANS turbulence (Feature 10)
    "KOmegaSSTResiduals", "SpalartAllmarasResiduals", "get_rans_preset",
    # CSG geometry (Feature 12)
    "CSGRectangle", "CSGCircle", "CSGEllipse", "CSGPolygon",
    "CSGUnion", "CSGIntersection", "CSGDifference",
    "lshape", "csg_annulus", "channel_with_hole", "t_junction",
    # Post-processing viz (Feature 17)
    "FlowVisualizer", "compute_streamlines", "compute_isosurface",
    "plot_streamlines_2d_model", "plot_isosurface_3d", "plot_volume_slice",
    # Multi-backend (Feature 15)
    "get_backend", "set_backend", "Backend", "JAXBackend", "jit_pinn", "vmap_residual",
    # Dynamics (Feature 18)
    "RigidBody", "RigidBodySystem", "RigidBodyState",
    "MPMSimulator", "MPMState",
    "SPHParticles", "ParticleSystem",
    # World model (Feature 19)
    "CosmosAdapter", "WorldModelConfig",
    "PhysicsVideoDataset", "SimToRealAdapter",
    "PhysicalScene", "SceneObject",
    # Adjoint shape opt (Feature 16)
    "ContinuousAdjointSolver", "ShapeParametrization",
    "DragAdjointObjective", "naca_parametric",
    # CAD → CFD (Feature 20)
    "CFDMesh", "NSFlowSolver", "CADToCFDPipeline",
    # Lazy submodule aliases (v0.5)
    "symbolic", "backend", "dynamics", "worldmodel",
]
