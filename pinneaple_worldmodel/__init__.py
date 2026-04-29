"""World foundation model integration for PINNeAPPle physics simulation.

Provides adapters to:

1. Use NVIDIA Cosmos as a physics prior / world model for conditioning PINN
   training with video-generation priors.
2. Generate synthetic training data from physics-based video generation
   (advection-diffusion fallback when Cosmos is unavailable).
3. Bridge the sim-to-real gap via domain randomisation + world-model alignment.

Quick start::

    from pinneaple_worldmodel import CosmosAdapter, WorldModelConfig, PhysicsVideoDataset

    # Create adapter (uses physics fallback if Cosmos is not installed)
    adapter = CosmosAdapter(WorldModelConfig(n_frames=8))

    # Generate physics video from an initial state tensor
    import torch
    initial_state = torch.rand(3, 64, 64)
    frames = adapter.generate(initial_state, "turbulent channel flow", n_frames=8)
    # frames: (8, 3, 64, 64)

    # Extract physical state proxies
    state = adapter.extract_state(frames)
    # state: {"velocity_proxy": ..., "pressure_proxy": ...}

    # Dataset for PINN training
    dataset = PhysicsVideoDataset("path/to/videos", field_names=["u", "v", "p"])

    # Scene description
    from pinneaple_worldmodel import PhysicalScene, SceneObject
    scene = PhysicalScene(description="flow past cylinder")
    scene.add_object(SceneObject("cylinder", position=(0.5, 0.5, 0.0)))
    x_col = scene.sample_collocation_points(n=1000)
"""

from .adapter import CosmosAdapter, WorldModelConfig, SimToRealAdapter
from .data_gen import PhysicsVideoDataset
from .scene import PhysicalScene, SceneObject

__all__ = [
    # Cosmos adapter
    "CosmosAdapter",
    "WorldModelConfig",
    "SimToRealAdapter",
    # Data generation
    "PhysicsVideoDataset",
    # Scene representation
    "PhysicalScene",
    "SceneObject",
]
