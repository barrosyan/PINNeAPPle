from __future__ import annotations

"""Tasks (benchmarks) available in :mod:`pinneaple_arena`."""

from .base import ArenaTask, TaskResult
from .flow_obstacle_2d import FlowObstacle2DTask
from .lid_driven_cavity_3d import LidDrivenCavity3DTask

# Benchmark tasks (multi-architecture evaluation)
from .burgers_1d import Burgers1DTask
from .poisson_2d import Poisson2DTask
from .heat_1d import Heat1DTask
from .wave_1d import Wave1DTask
from .allen_cahn_1d import AllenCahn1DTask
from .navier_stokes_tgv_2d import NavierStokesTGV2DTask

__all__ = [
    "ArenaTask",
    "TaskResult",
    "FlowObstacle2DTask",
    "LidDrivenCavity3DTask",
    # Benchmark tasks
    "Burgers1DTask",
    "Poisson2DTask",
    "Heat1DTask",
    "Wave1DTask",
    "AllenCahn1DTask",
    "NavierStokesTGV2DTask",
]