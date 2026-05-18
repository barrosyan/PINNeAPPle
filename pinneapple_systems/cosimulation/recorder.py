"""Trajectory recording for co-simulation runs.

Usage::

    recorder = TrajectoryRecorder()
    recorder.watch("mass", "x")       # watch port 'x' on node 'mass'
    recorder.watch("mass", "v")
    recorder.watch_node("spring", ["F", "x"])

    # During simulation (called by CoSimEngine):
    recorder.record(t, "mass", {"x": tensor, "v": tensor})

    # After run:
    traj = recorder.get("mass", "x")
    plt.plot(traj.times, traj.values[:, 0])
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Trajectory — single (node, port) signal
# ---------------------------------------------------------------------------

class Trajectory:
    """Time series for a single (node, port) signal.

    Values are stored as detached CPU tensors; ``times`` and ``values``
    properties expose them as NumPy arrays for easy plotting.
    """

    def __init__(self, node: str, port: str) -> None:
        self.node = node
        self.port = port
        self._times: List[float] = []
        self._values: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def record(self, t: float, value: torch.Tensor) -> None:
        self._times.append(float(t))
        self._values.append(value.detach().cpu())

    # ------------------------------------------------------------------
    @property
    def times(self) -> np.ndarray:
        """1-D array of recorded time stamps."""
        return np.array(self._times, dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        """Array of shape (T, *value_shape) with all recorded values."""
        if not self._values:
            return np.array([])
        stacked = torch.stack(self._values)          # (T, ...)
        return stacked.numpy()

    def tensor(self) -> torch.Tensor:
        """Return values as a float32 tensor of shape (T, *value_shape)."""
        if not self._values:
            return torch.empty(0)
        return torch.stack(self._values).float()

    def __len__(self) -> int:
        return len(self._times)

    def __repr__(self) -> str:
        return (
            f"Trajectory({self.node}.{self.port}, "
            f"steps={len(self)}, "
            f"shape={tuple(self._values[0].shape) if self._values else ()})"
        )


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Records time-stamped port outputs from multiple nodes.

    Only ports explicitly registered with ``watch()`` / ``watch_node()``
    are stored — everything else is ignored to keep memory usage bounded.
    """

    def __init__(self) -> None:
        self._watched: Dict[str, List[str]] = defaultdict(list)
        self._data: Dict[Tuple[str, str], Trajectory] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def watch(self, node: str, port: str) -> "TrajectoryRecorder":
        """Register a single (node, port) pair for recording."""
        if port not in self._watched[node]:
            self._watched[node].append(port)
        key = (node, port)
        if key not in self._data:
            self._data[key] = Trajectory(node, port)
        return self

    def watch_node(self, node: str, ports: List[str]) -> "TrajectoryRecorder":
        """Register multiple ports of the same node at once."""
        for p in ports:
            self.watch(node, p)
        return self

    # ------------------------------------------------------------------
    # Recording (called by CoSimEngine)
    # ------------------------------------------------------------------

    def record(
        self,
        t: float,
        node_name: str,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Store current outputs for watched ports of *node_name* at time *t*."""
        for port in self._watched.get(node_name, []):
            if port in outputs:
                self._data[(node_name, port)].record(t, outputs[port])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, node: str, port: str) -> Trajectory:
        """Return the Trajectory for (node, port).

        Raises:
            KeyError: if the pair was never watched.
        """
        key = (node, port)
        if key not in self._data:
            raise KeyError(
                f"No trajectory for {node}.{port}. "
                f"Call watch({node!r}, {port!r}) before running the simulation."
            )
        return self._data[key]

    def all_trajectories(self) -> List[Trajectory]:
        """Return all recorded Trajectory objects."""
        return list(self._data.values())

    def summary(self) -> Dict[str, int]:
        """Return {node.port: number_of_recorded_steps} for all watched pairs."""
        return {
            f"{node}.{port}": len(self._data[(node, port)])
            for node, ports in self._watched.items()
            for port in ports
            if (node, port) in self._data
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> "TrajectoryRecorder":
        """Clear all recorded data while keeping the watch list intact."""
        for key in list(self._data.keys()):
            node, port = key
            self._data[key] = Trajectory(node, port)
        return self

    def __repr__(self) -> str:
        watched = [
            f"{n}.{p}"
            for n, ps in self._watched.items()
            for p in ps
        ]
        return f"TrajectoryRecorder(watching={watched})"
