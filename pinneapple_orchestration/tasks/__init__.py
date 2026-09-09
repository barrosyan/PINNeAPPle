from __future__ import annotations

from .data_generation_tasks import generate_scenarios_task
from .training_tasks import train_component_task
from .benchmark_tasks import benchmark_component_type_task
from .registry_tasks import publish_model_task
from .digital_twin_tasks import prepare_digital_twin_task

__all__ = [
    "generate_scenarios_task",
    "train_component_task",
    "benchmark_component_type_task",
    "publish_model_task",
    "prepare_digital_twin_task",
]
