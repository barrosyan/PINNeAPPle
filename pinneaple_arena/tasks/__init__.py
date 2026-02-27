from __future__ import annotations

"""Tasks (benchmarks) available in :mod:`pinneaple_arena`."""

from .base import ArenaTask, TaskResult
from .flow_obstacle_2d import FlowObstacle2DTask

__all__ = ["ArenaTask", "TaskResult", "FlowObstacle2DTask"]