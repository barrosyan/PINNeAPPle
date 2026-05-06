"""pinneaple_worldmodel — Physics AI World Model Pipeline.

Generates training datasets for a physics world model (an AGI for physics)
using all simulation, validation, and modelling tools available in Pinneaple.

What is a physics world model?
-------------------------------
A model that learns to predict physical system evolution:

    f_θ(state_t, params) → state_{t+1}

where ``state_t`` is a spatial field (temperature, velocity, pressure, …) and
``params`` encodes the PDE parameters (diffusivity, Reynolds number, …).
When trained across many physics domains, the model builds a general prior
over physical system dynamics — a foundation model for physics AI.

Pipeline overview
-----------------
::

    PhysicsScenario (define PDEs + parameter ranges)
          ↓
    PhysicsSimulator (generate trajectories via pinneaple_solvers)
          ↓
    WorldModelDataset (format as (state_t, params) → state_{t+1})
          ↓
    PhysicsWorldModel (FNO-based: learns the state-transition operator)
          ↓
    WorldModelTrainer (rollout loss + optional physics consistency)
          ↓
    PhysicsCurriculum (staged training: easy → hard physics)
          ↓
    WorldModelPipeline (single entry point for the full flow)

Built-in physics scenarios
--------------------------
``heat_2d``, ``burgers_1d``, ``wave_1d``, ``advection_2d``,
``ns2d_cavity``, ``heat_multiscale`` — see :data:`BUILTIN_SCENARIOS`.

Quick start
-----------
::

    from pinneaple_worldmodel import WorldModelPipeline, PipelineConfig

    pipeline = WorldModelPipeline(PipelineConfig(
        scenarios=["heat_2d", "burgers_1d", "advection_2d"],
        n_samples_per_scenario=500,
        epochs=100,
        device="cuda",          # or "cpu"
        save_dir="./wm_output",
    ))
    model, history = pipeline.run()

    # Rollout the trained model on a new initial condition
    import torch
    state_0 = torch.randn(1, 1, 64, 64)            # (B, C, H, W)
    context  = torch.zeros(1, model.config.context_dim)
    with torch.no_grad():
        future_states = model.rollout(state_0, context, n_steps=20)
    # future_states : (1, 20, 1, 64, 64)

Curriculum training
-------------------
::

    from pinneaple_worldmodel import PhysicsCurriculum, CurriculumConfig

    # Staged: heat → burgers/wave → advection 2D → Navier-Stokes → high-res
    model = PhysicsCurriculum(CurriculumConfig(device="cuda")).run()

Custom scenario
---------------
::

    from pinneaple_worldmodel import PhysicsScenario, DatasetBuilder, DatasetConfig

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
"""
from __future__ import annotations

from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .simulator import PhysicsSimulator, TrajectoryData
from .dataset import WorldModelDataset, DatasetBuilder, DatasetConfig
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig, WorldModelLoss
from .curriculum import PhysicsCurriculum, CurriculumConfig, CurriculumStage
from .pipeline import WorldModelPipeline, PipelineConfig

__all__ = [
    # Scenarios
    "PhysicsScenario",
    "BUILTIN_SCENARIOS",
    # Simulator
    "PhysicsSimulator",
    "TrajectoryData",
    # Dataset
    "WorldModelDataset",
    "DatasetBuilder",
    "DatasetConfig",
    # Model
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
    # Pipeline
    "WorldModelPipeline",
    "PipelineConfig",
]
