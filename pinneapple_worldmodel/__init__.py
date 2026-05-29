"""pinneapple_worldmodel — Generalist Physics AI World Model.

Builds a *Physics Foundation Model* — a generalist AI trained across many
physics domains — using all simulation, validation, meta-learning, and
modelling tools available in Pinneapple.

What is a Physics Foundation Model?
------------------------------------
A model that learns to predict physical system evolution::

    f_θ(state_t, descriptor) → state_{t+1}

where ``state_t`` is a spatial field (temperature, velocity, pressure, …) and
``descriptor`` encodes the PDE parameters, domain shape, and physics context.
Trained across many physics domains, the model builds a general prior over
physical system dynamics — a foundation model for physics AI.

Pipeline overview
-----------------
::

    PhysicsAIPipeline
    │
    ├─ Stage 1 — Multi-source dataset generation
    │     PhysicsDatasetFactory
    │       ├─ solver  → pinneapple_solvers (FDM / LBM / FEM / SPH)
    │       ├─ pinn    → pinneapple_pinn (PINN residual data)
    │       ├─ symbolic→ pinneapple_symbolic (analytical solutions)
    │       └─ colloc  → pinneapple_data (collocation sampling)
    │     DatasetCatalog (organised by scenario + source)
    │
    ├─ Stage 2 — Specialist training
    │     SpecialistTrainer
    │       ├─ pinneapple_train  (AdamW, AMP, cosine LR)
    │       ├─ pinneapple_validate (conservation checks)
    │       ├─ pinneapple_uq     (aleatoric + epistemic UQ)
    │       └─ pinneapple_transfer (domain adaptation)
    │     ModelZoo (one specialist per scenario)
    │
    ├─ Stage 3 — Meta-learning
    │     MetaLearner (MAML / Reptile, or pinneapple_meta)
    │     → meta-initialised PhysicsWorldModel
    │
    ├─ Stage 4 — Foundation model assembly
    │     WaMaModel (weight-averaged soup of specialists)
    │     PhysicsFoundationModel (FNO + cross-attention + LoRA adapters)
    │     Fine-tuned on merged multi-source catalog
    │
    └─ Stage 5 — Benchmark evaluation
          PhysicsBenchmark (6 standard tasks, conservation checks)

Secondary pipeline — solve any physics problem
-----------------------------------------------
::

    PhysicsOrchestrator + ProblemStatement

Treats every Pinneapple capability as a callable tool and chains them
automatically to solve forward / inverse / design / discovery /
forecast / digital-twin / uncertainty problems.

Quick start
-----------
::

    from pinneapple_worldmodel import PhysicsAIPipeline, PhysicsAIConfig

    pipeline = PhysicsAIPipeline(PhysicsAIConfig(
        scenarios=["heat_2d", "burgers_1d", "advection_2d", "ns2d_cavity"],
        sources=["solver", "pinn"],
        n_samples=500,
        device="cuda",
        save_dir="./physics_ai_output",
    ))
    result = pipeline.run()
    mega_model = result.mega_model
    zoo        = result.zoo

Orchestrator quick start
------------------------
::

    from pinneapple_worldmodel import PhysicsOrchestrator, ProblemStatement

    # Solve an inverse problem
    result = PhysicsOrchestrator().solve(ProblemStatement(
        kind="inverse",
        pde_hint="burgers_1d",
        observations=my_data,
        output=["params_estimate", "uncertainty"],
    ))

    # Discover governing equations from data
    result = PhysicsOrchestrator().solve(ProblemStatement(
        kind="discovery",
        observations=trajectory_data,
        output=["equations"],
    ))

Custom scenario
---------------
::

    from pinneapple_worldmodel import PhysicsScenario, DatasetBuilder, DatasetConfig

    my_scenario = PhysicsScenario(
        name="my_ns",
        pde_kind="ns2d",
        grid_shape=(128, 128),
        t_span=(0.0, 10.0),
        n_steps=100,
        param_ranges={"Re": (200.0, 2000.0)},
        bc_type="dirichlet_zero",
    )
    dataset = DatasetBuilder(DatasetConfig(
        scenarios=[my_scenario],
        n_samples_per_scenario=300,
    )).build()

Tool registry
-------------
::

    from pinneapple_worldmodel import PhysicsToolRegistry

    reg = PhysicsToolRegistry()
    reg.register_all()
    print(reg.summary())
    tool = reg.get("simulate_trajectory")
    traj = tool.call(scenario="heat_2d", n_steps=50)
"""
from __future__ import annotations

# --- Core scenario & simulation ---
from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .simulator import PhysicsSimulator, TrajectoryData

# --- Dataset ---
from .dataset import WorldModelDataset, DatasetBuilder, DatasetConfig

# --- Multi-source dataset factory ---
from .dataset_factory import (
    PhysicsDatasetFactory,
    FactoryConfig,
    DatasetCatalog,
    DatasetEntry,
)

# --- Geometry ---
from .geometry import (
    GeometryBase,
    Rectangle,
    Circle,
    Polygon,
    Box3D,
    Sphere,
    Union,
    Intersection,
    Difference,
    BoundaryRegion,
    PhysicsDomain,
    make_unit_square,
    make_cavity,
    make_channel,
    make_channel_with_cylinder,
    make_l_shaped,
    make_annulus,
    BUILTIN_DOMAINS,
)

# --- World model (FNO-based) ---
from .model import PhysicsWorldModel, WorldModelConfig

# --- Trainer ---
from .trainer import WorldModelTrainer, WorldModelTrainConfig, WorldModelLoss

# --- Curriculum (legacy) ---
from .curriculum import PhysicsCurriculum, CurriculumConfig, CurriculumStage

# --- Model zoo ---
from .model_zoo import ModelZoo, ZooEntry, EnsembleModel, WaMaModel

# --- Specialist trainer ---
from .specialist_trainer import SpecialistTrainer, SpecialistConfig

# --- Meta-learning ---
from .meta_learning import MetaLearner, MetaConfig, TaskDistribution

# --- Foundation model ---
from .mega_model import (
    PhysicsFoundationModel,
    FoundationConfig,
    LoRALinear,
    PhysicsDescriptorEncoder,
)

# --- Benchmark ---
from .benchmark import (
    PhysicsBenchmark,
    BenchmarkTask,
    BenchmarkResult,
    BUILTIN_TASKS,
)

# --- Tool registry ---
from .physics_tools import PhysicsToolRegistry, PhysicsTool

# --- Orchestrator ---
from .orchestrator import PhysicsOrchestrator, ProblemStatement, OrchestratorResult

# --- Pipeline ---
from .pipeline import PhysicsAIPipeline, PhysicsAIConfig, PhysicsAIPipelineResult

# Legacy pipeline (kept for backwards compatibility)
from .pipeline import PhysicsAIPipeline as WorldModelPipeline  # noqa: F401

# --- Training & dataset generation entry points ---
from .train import (
    train_specialist,
    train_meta,
    train_foundation,
    run_pipeline,
)
from .generate_datasets import (
    generate as generate_datasets,
    generate_via_factory,
    load_catalog,
)

__all__ = [
    # Scenario
    "PhysicsScenario",
    "BUILTIN_SCENARIOS",
    # Simulation
    "PhysicsSimulator",
    "TrajectoryData",
    # Dataset
    "WorldModelDataset",
    "DatasetBuilder",
    "DatasetConfig",
    # Dataset factory
    "PhysicsDatasetFactory",
    "FactoryConfig",
    "DatasetCatalog",
    "DatasetEntry",
    # Geometry
    "GeometryBase",
    "Rectangle",
    "Circle",
    "Polygon",
    "Box3D",
    "Sphere",
    "Union",
    "Intersection",
    "Difference",
    "BoundaryRegion",
    "PhysicsDomain",
    "make_unit_square",
    "make_cavity",
    "make_channel",
    "make_channel_with_cylinder",
    "make_l_shaped",
    "make_annulus",
    "BUILTIN_DOMAINS",
    # World model
    "PhysicsWorldModel",
    "WorldModelConfig",
    # Trainer
    "WorldModelTrainer",
    "WorldModelTrainConfig",
    "WorldModelLoss",
    # Curriculum
    "PhysicsCurriculum",
    "CurriculumConfig",
    "CurriculumStage",
    # Model zoo
    "ModelZoo",
    "ZooEntry",
    "EnsembleModel",
    "WaMaModel",
    # Specialist trainer
    "SpecialistTrainer",
    "SpecialistConfig",
    # Meta-learning
    "MetaLearner",
    "MetaConfig",
    "TaskDistribution",
    # Foundation model
    "PhysicsFoundationModel",
    "FoundationConfig",
    "LoRALinear",
    "PhysicsDescriptorEncoder",
    # Benchmark
    "PhysicsBenchmark",
    "BenchmarkTask",
    "BenchmarkResult",
    "BUILTIN_TASKS",
    # Tool registry
    "PhysicsToolRegistry",
    "PhysicsTool",
    # Orchestrator
    "PhysicsOrchestrator",
    "ProblemStatement",
    "OrchestratorResult",
    # Pipeline
    "PhysicsAIPipeline",
    "PhysicsAIConfig",
    "PhysicsAIPipelineResult",
    # Training entry points
    "train_specialist",
    "train_meta",
    "train_foundation",
    "run_pipeline",
    # Dataset generation entry points
    "generate_datasets",
    "generate_via_factory",
    "load_catalog",
]
