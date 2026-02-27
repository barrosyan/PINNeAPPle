from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pinneaple_arena.bundle.loader import BundleData


@dataclass(frozen=True)
class TaskResult:
    """Standard output for a task evaluation."""
    metrics: Dict[str, float]
    artifacts: Optional[Dict[str, Any]] = None


class ArenaTask:
    """Base class for Arena tasks."""
    task_id: str = "base"

    def compute_metrics(self, bundle: BundleData, backend_outputs: Dict[str, Any]) -> Dict[str, float] | TaskResult:
        raise NotImplementedError