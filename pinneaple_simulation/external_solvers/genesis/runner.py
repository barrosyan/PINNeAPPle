"""Genesis AI simulation runner for PINNeAPPle.

GenesisRunner drives the scene.step() loop and snapshots entity states
(position, velocity) at each timestep into NumPy/PyTorch arrays.

Supported state extraction strategies
--------------------------------------
- "rigid"  : reads entity.get_state() for rigid bodies
- "mpm"    : reads particle positions from MPM entities
- "sph"    : reads particle positions from SPH liquid entities
- "custom" : caller provides a state_fn(scene, entities) -> dict hook
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .scene_builder import GenesisConfig, build_scene


class GenesisRunner:
    """Execute a Genesis AI scene and collect trajectory observations.

    Parameters
    ----------
    config : GenesisConfig
    state_fn : optional callable (scene, entity_map) -> dict that returns
        a dict of {field_name: array}. If None, default rigid-state
        extraction is used for all entities.
    controller_fn : optional callable (scene, entity_map, step_idx) -> None
        invoked before each scene.step() — use for robot control injection.
    """

    def __init__(
        self,
        config: GenesisConfig,
        state_fn: Optional[Callable] = None,
        controller_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.state_fn = state_fn
        self.controller_fn = controller_fn
        self._scene = None

    @property
    def scene(self):
        if self._scene is None:
            self._scene = build_scene(self.config)
        return self._scene

    @property
    def entities(self) -> dict:
        return self.scene._pinneaple_entities  # type: ignore[attr-defined]

    def reset(self) -> None:
        self.scene.reset()

    def _default_state(self, scene, entity_map: dict) -> dict:
        """Extract rigid-body state (pos, vel) for all entities."""
        records: dict = {}
        for name, entity in entity_map.items():
            try:
                state = entity.get_state()
                pos = np.asarray(state.pos)
                vel = np.asarray(state.vel) if hasattr(state, "vel") else np.zeros_like(pos)
                records[f"{name}_pos"] = pos
                records[f"{name}_vel"] = vel
            except Exception:
                pass
        return records

    def simulate(self) -> dict:
        """Run the simulation and collect a trajectory.

        Returns
        -------
        dict mapping field names → (T, ...) NumPy arrays.
        An extra key "step" is included with integer step indices.
        """
        cfg = self.config
        scene = self.scene
        entity_map = self.entities
        self.reset()

        buffers: Optional[Dict[str, list]] = None

        for step_idx in range(cfg.n_steps):
            if self.controller_fn is not None:
                self.controller_fn(scene, entity_map, step_idx)

            scene.step()

            if self.state_fn is not None:
                snapshot = self.state_fn(scene, entity_map)
            else:
                snapshot = self._default_state(scene, entity_map)
            snapshot["step"] = np.array(step_idx)

            if buffers is None:
                buffers = {k: [] for k in snapshot}
            for k, v in snapshot.items():
                buffers[k].append(np.asarray(v))

        if buffers is None:
            return {}
        return {k: np.stack(v, axis=0) for k, v in buffers.items()}

    def collect_trajectories(
        self, n_rollouts: int, reset_between: bool = True
    ) -> List[dict]:
        """Run multiple rollouts.

        Parameters
        ----------
        n_rollouts : number of rollouts
        reset_between : call self.reset() before each rollout
        """
        trajectories = []
        for _ in range(n_rollouts):
            if reset_between:
                self.reset()
            trajectories.append(self.simulate())
        return trajectories
