"""MuJoCo integration for PINNeAPPle.

Provides a thin workflow layer over the MuJoCo Python bindings
(google-deepmind/mujoco) with structured configuration, trajectory
collection, and conversion to PINNeAPPle's UPD (PhysicalSample) format.

Sub-modules
-----------
loader      — MuJoCoConfig dataclass, load_model(), make_data()
runner      — MuJoCoRunner (step loop, multi-rollout collection)
adapter     — trajectory_to_upd(), trajectories_to_upd()
mjx_backend — MJXParallelRunner (JAX/GPU vectorised rollouts via MJX)

Quick start
-----------
>>> from pinneapple_simulation.external_solvers.mujoco import (
...     MuJoCoConfig, MuJoCoRunner, trajectory_to_upd
... )
>>> cfg = MuJoCoConfig(model_path="robot.xml", n_steps=500, dt=0.002)
>>> runner = MuJoCoRunner(cfg)
>>> traj = runner.simulate()          # dict of numpy arrays
>>> sample = trajectory_to_upd(traj)  # → PhysicalSample (UPD)

Parallel rollouts (requires JAX + CUDA):
>>> from pinneapple_simulation.external_solvers.mujoco import MJXParallelRunner
>>> batch_runner = MJXParallelRunner(cfg, n_envs=1024)
>>> samples = batch_runner.rollout_to_upd()

PINN-in-the-loop controller injection:
>>> def my_ctrl(model, data):
...     data.ctrl[:] = pinn_policy(data.qpos, data.qvel)
>>> runner = MuJoCoRunner(cfg, controller=my_ctrl)
>>> traj = runner.simulate()
"""
from .loader import MuJoCoConfig, load_model, make_data
from .runner import MuJoCoRunner
from .adapter import trajectory_to_upd, trajectories_to_upd

try:
    from .mjx_backend import MJXParallelRunner
except Exception:
    MJXParallelRunner = None  # type: ignore

__all__ = [
    "MuJoCoConfig",
    "load_model",
    "make_data",
    "MuJoCoRunner",
    "trajectory_to_upd",
    "trajectories_to_upd",
    "MJXParallelRunner",
]
