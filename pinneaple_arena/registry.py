from __future__ import annotations

"""Registries for tasks and backends.

The arena started as an MVP with hard-coded task/backend selection.
This module makes the system extensible while keeping simple APIs.
"""

from typing import Dict, List, Type, TYPE_CHECKING

# -----------------------------
# Tasks
# -----------------------------

TASK_REGISTRY: Dict[str, Type["ArenaTask"]] = {}


def register_task(task_cls: Type["ArenaTask"]) -> Type["ArenaTask"]:
    """Decorator to register a task by its ``task_id``."""
    task_id = getattr(task_cls, "task_id", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{task_cls.__name__} must define a non-empty 'task_id' string")

    existing = TASK_REGISTRY.get(task_id)
    if existing is not None and existing is not task_cls:
        raise ValueError(f"Duplicate task_id='{task_id}' registered by {existing.__name__} and {task_cls.__name__}")

    TASK_REGISTRY[task_id] = task_cls
    return task_cls


def get_task(task_id: str) -> Type["ArenaTask"]:
    if task_id not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task_id='{task_id}'. Available: [{available}]")
    return TASK_REGISTRY[task_id]


def list_tasks() -> List[str]:
    return sorted(TASK_REGISTRY.keys())


# -----------------------------
# Backends
# -----------------------------

BACKEND_REGISTRY: Dict[str, Type["Backend"]] = {}


def register_backend(backend_cls: Type["Backend"]) -> Type["Backend"]:
    """Decorator to register a backend by its ``name``."""
    name = getattr(backend_cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{backend_cls.__name__} must define a non-empty 'name' string")

    existing = BACKEND_REGISTRY.get(name)
    if existing is not None and existing is not backend_cls:
        raise ValueError(f"Duplicate backend name='{name}' registered by {existing.__name__} and {backend_cls.__name__}")

    BACKEND_REGISTRY[name] = backend_cls
    return backend_cls


def get_backend(name: str) -> Type["Backend"]:
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise KeyError(f"Unknown backend='{name}'. Available: [{available}]")
    return BACKEND_REGISTRY[name]


def list_backends() -> List[str]:
    return sorted(BACKEND_REGISTRY.keys())


if TYPE_CHECKING:  # pragma: no cover
    from pinneaple_arena.tasks.base import ArenaTask
    from pinneaple_arena.backends.base import Backend