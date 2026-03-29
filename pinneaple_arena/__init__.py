from __future__ import annotations

from . import io, bundle, tasks, backends, runner
from .api import Arena, ArenaResult, ArenaCompareResult
from .runner.run_arena_yaml import run_arena_experiment

try:
    from .runner.run_pipeline import run_full_pipeline
except Exception:
    pass
from .registry import (
    TASK_REGISTRY,
    BACKEND_REGISTRY,
    register_task,
    register_backend,
    get_task,
    get_backend,
    list_tasks,
    list_backends,
)

__all__ = [
    "Arena",
    "ArenaResult",
    "ArenaCompareResult",
    "io",
    "bundle",
    "tasks",
    "backends",
    "runner",
    "TASK_REGISTRY",
    "BACKEND_REGISTRY",
    "register_task",
    "register_backend",
    "get_task",
    "get_backend",
    "list_tasks",
    "list_backends",
    "run_arena_experiment",
    "run_full_pipeline",
]