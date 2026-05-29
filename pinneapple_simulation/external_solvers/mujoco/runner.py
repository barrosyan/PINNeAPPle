"""MuJoCo simulation runner for PINNeAPPle.

MuJoCoRunner executes the mj_step loop and collects trajectory data
(qpos, qvel, ctrl, xpos, sensordata, time) into NumPy arrays suitable
for surrogate / PINN training.

Controller injection
--------------------
Pass a callable controller(model, data) -> None that writes data.ctrl
before each step. This lets you close the loop with a PINN-predicted
or RL-policy-driven controller:

    def my_ctrl(model, data):
        data.ctrl[:] = pinn_policy(data.qpos, data.qvel)

    runner = MuJoCoRunner(config, controller=my_ctrl)
    traj = runner.simulate()
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .loader import MuJoCoConfig, load_model, make_data


class MuJoCoRunner:
    """Run a MuJoCo model and collect a trajectory.

    Parameters
    ----------
    config : MuJoCoConfig
    controller : optional callable (model, data) -> None executed before
        every batch of n_substeps steps.
    """

    def __init__(
        self,
        config: MuJoCoConfig,
        controller: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.controller = controller
        self._model = None
        self._data = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.config)
        return self._model

    @property
    def data(self):
        if self._data is None:
            self._data = make_data(self.model)
        return self._data

    def reset(self) -> None:
        """Reset simulation to initial state."""
        import mujoco
        mujoco.mj_resetData(self.model, self.data)

    def simulate(self) -> dict:
        """Run the full simulation and return collected arrays.

        Returns
        -------
        dict with keys:
          time     : (T,) float64 — simulation timestamps
          qpos     : (T, nq) float64 — generalized positions
          qvel     : (T, nv) float64 — generalized velocities
          ctrl     : (T, nu) float64 — actuator controls
          xpos     : (T, nbody, 3) float64 — Cartesian body positions
          sensor   : (T, nsensor) float64 — sensor readings (may be empty)
        """
        import mujoco

        cfg = self.config
        model = self.model
        data = self.data
        self.reset()

        n_records = cfg.n_steps
        nq = model.nq
        nv = model.nv
        nu = model.nu
        nbody = model.nbody
        nsensor = model.nsensor

        time_buf = np.empty(n_records, dtype=np.float64)
        qpos_buf = np.empty((n_records, nq), dtype=np.float64)
        qvel_buf = np.empty((n_records, nv), dtype=np.float64)
        ctrl_buf = np.empty((n_records, nu), dtype=np.float64)
        xpos_buf = np.empty((n_records, nbody, 3), dtype=np.float64)
        sensor_buf = np.empty((n_records, nsensor), dtype=np.float64)

        step_idx = 0
        while step_idx < n_records:
            if self.controller is not None:
                self.controller(model, data)
            mujoco.mj_step(model, data, nstep=cfg.n_substeps)

            time_buf[step_idx] = data.time
            qpos_buf[step_idx] = data.qpos
            qvel_buf[step_idx] = data.qvel
            ctrl_buf[step_idx] = data.ctrl
            xpos_buf[step_idx] = data.xpos.reshape(nbody, 3)
            if nsensor > 0:
                sensor_buf[step_idx] = data.sensordata
            step_idx += 1

        return {
            "time": time_buf,
            "qpos": qpos_buf,
            "qvel": qvel_buf,
            "ctrl": ctrl_buf,
            "xpos": xpos_buf,
            "sensor": sensor_buf,
        }

    def collect_trajectories(self, n_rollouts: int, reset_between: bool = True) -> list[dict]:
        """Run multiple rollouts and return a list of trajectory dicts.

        Parameters
        ----------
        n_rollouts : number of independent rollouts
        reset_between : reset to initial state before each rollout
        """
        trajectories = []
        for _ in range(n_rollouts):
            if reset_between:
                self.reset()
            trajectories.append(self.simulate())
        return trajectories
