"""Genesis AI integration for PINNeAPPle.

Provides a structured workflow over the genesis-world physics engine
(Genesis-Embodied-AI/Genesis) for robotics, embodied AI, and multi-physics
simulation, with output conversion to PINNeAPPle's UPD (PhysicalSample) format.

Sub-modules
-----------
scene_builder — GenesisConfig, EntitySpec, build_scene()
runner        — GenesisRunner (step loop, trajectory collection)
adapter       — genesis_traj_to_upd(), genesis_trajs_to_upd()

Quick start — rigid body simulation
-------------------------------------
>>> from pinneaple_simulation.external_solvers.genesis import (
...     GenesisConfig, EntitySpec, GenesisRunner, genesis_traj_to_upd
... )
>>> cfg = GenesisConfig(
...     backend="gpu",
...     n_steps=500,
...     entities=[
...         EntitySpec(morph={"type": "Plane"}),
...         EntitySpec(
...             morph={"type": "URDF", "file": "robot.urdf", "fixed": True},
...             material={"type": "Rigid"},
...             name="robot",
...         ),
...     ],
... )
>>> runner = GenesisRunner(cfg)
>>> traj = runner.simulate()              # dict of numpy arrays
>>> sample = genesis_traj_to_upd(traj)   # → PhysicalSample (UPD)

Quick start — MPM soft-body with differentiable simulation
-----------------------------------------------------------
>>> cfg = GenesisConfig(
...     backend="gpu",
...     requires_grad=True,
...     mpm_options={"lower_bound": (-1, -1, 0), "upper_bound": (1, 1, 2)},
...     entities=[
...         EntitySpec(morph={"type": "Sphere", "radius": 0.1}, material={"type": "MPM.Elastic", "rho": 500}),
...     ],
... )

Parallel environments (RL / dataset generation)
------------------------------------------------
>>> cfg = GenesisConfig(backend="gpu", n_envs=512, n_steps=200, entities=[...])
>>> runner = GenesisRunner(cfg)
>>> trajs = runner.collect_trajectories(n_rollouts=1)  # each step batched over 512 envs

Custom controller injection (PINN-in-the-loop)
----------------------------------------------
>>> def ctrl(scene, entities, step_idx):
...     robot = entities["robot"]
...     robot.control_dofs_position(pinn_policy(step_idx))
>>> runner = GenesisRunner(cfg, controller_fn=ctrl)
"""
from .scene_builder import GenesisConfig, EntitySpec, build_scene
from .runner import GenesisRunner
from .adapter import genesis_traj_to_upd, genesis_trajs_to_upd

__all__ = [
    "GenesisConfig",
    "EntitySpec",
    "build_scene",
    "GenesisRunner",
    "genesis_traj_to_upd",
    "genesis_trajs_to_upd",
]
