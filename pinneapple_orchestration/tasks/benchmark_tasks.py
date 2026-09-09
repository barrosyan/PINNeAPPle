"""pinneapple_orchestration.tasks.benchmark_tasks — the task behind
flow_03_benchmark. Thin wrapper over
``pinneapple_systems.component_library.benchmark_component_type`` — no
benchmarking logic lives here.
"""
from __future__ import annotations

from typing import Any, Optional

from pinneapple_systems.component_library import ComponentBenchmarkResult, benchmark_component_type

from .._prefect_compat import task


@task
def benchmark_component_type_task(
    component_type: str,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
) -> ComponentBenchmarkResult:
    return benchmark_component_type(component_type, train_loader, val_loader, epochs=epochs, lr=lr)
