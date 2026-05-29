"""MJX (JAX/GPU) backend for massively parallel MuJoCo rollouts.

MJX uploads a MjModel to JAX device memory and runs thousands of
environments simultaneously using jax.vmap. This is ideal for generating
large training datasets for surrogate models and PINNs.

Requirements
------------
  pip install mujoco jax[cuda12]   # or jax[cpu] for CPU-only

Usage
-----
    from pinneapple_simulation.external_solvers.mujoco import MJXParallelRunner
    from pinneapple_simulation.external_solvers.mujoco import MuJoCoConfig

    cfg = MuJoCoConfig(model_path="robot.xml", n_steps=500)
    runner = MJXParallelRunner(cfg, n_envs=1024)
    # batched_traj shape: {qpos: (1024, 500, nq), ...}
    batched_traj = runner.rollout()
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .loader import MuJoCoConfig, load_model


class MJXParallelRunner:
    """Run N independent MuJoCo environments in parallel via MJX + JAX.vmap.

    Parameters
    ----------
    config : MuJoCoConfig
    n_envs : number of parallel environments
    device : JAX device string, e.g. "cuda:0", "cpu". None = JAX default.
    """

    def __init__(
        self,
        config: MuJoCoConfig,
        n_envs: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.n_envs = n_envs
        self.device = device
        self._mjx_model = None

    def _ensure_mjx(self):
        try:
            import jax
            from mujoco import mjx
        except ImportError:
            raise ImportError(
                "MJX requires JAX and MuJoCo. Install with:\n"
                "  pip install mujoco 'jax[cuda12]'"
            )
        if self._mjx_model is None:
            mj_model = load_model(self.config)
            self._mjx_model = mjx.put_model(mj_model)
        return self._mjx_model

    def rollout(self) -> dict:
        """Run batched rollout across n_envs environments.

        Returns
        -------
        dict with keys qpos, qvel, ctrl, xpos, time.
        Each value is a NumPy array of shape (n_envs, T, ...).
        """
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        mjx_model = self._ensure_mjx()
        n_envs = self.n_envs
        n_steps = self.config.n_steps

        def make_data(_):
            return mjx.make_data(mjx_model)

        init_data = jax.vmap(make_data)(jnp.arange(n_envs))

        def step_fn(data, _):
            data = mjx.step(mjx_model, data)
            record = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "ctrl": data.ctrl,
                "xpos": data.xpos,
                "time": data.time,
            }
            return data, record

        batched_step = jax.vmap(
            lambda d: jax.lax.scan(step_fn, d, None, length=n_steps)
        )

        _, traj = batched_step(init_data)

        return {k: np.asarray(v) for k, v in traj.items()}

    def rollout_to_upd(self) -> list:
        """Run batched rollout and return a list of PhysicalSamples (one per env)."""
        from .adapter import trajectory_to_upd

        batch = self.rollout()
        n_envs = self.n_envs
        samples = []
        for i in range(n_envs):
            env_traj = {k: v[i] for k, v in batch.items()}
            samples.append(
                trajectory_to_upd(env_traj, meta_extra={"env_idx": i, "backend": "mjx"})
            )
        return samples
