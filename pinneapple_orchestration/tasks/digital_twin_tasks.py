"""pinneapple_orchestration.tasks.digital_twin_tasks — the task behind
flow_05_digital_twin_prep.

Gated on stage: by default only a model already promoted to
``"production"`` in the registry (see ``registry_tasks.publish_model_task``)
can be prepped into a live ``DigitalTwin`` — mirroring the same
stage-promotion discipline the registry itself enforces, so a
still-``"development"``/``"staging"`` model can't accidentally end up
driving what looks like a production twin.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pinneapple_registry import ArtifactRegistry
from pinneapple_systems.digital_twin import DigitalTwin, DigitalTwinConfig

from .._prefect_compat import task


@task
def prepare_digital_twin_task(
    registry: ArtifactRegistry,
    problem_id: str,
    field_names: List[str],
    coord_names: Optional[List[str]] = None,
    *,
    version: Optional[str] = None,
    require_stage: Optional[str] = "production",
    model_cls: Optional[Any] = None,
    config: Optional[DigitalTwinConfig] = None,
    **init_kwargs: Any,
) -> DigitalTwin:
    v = version or registry.models.latest(problem_id)
    if v is None:
        raise KeyError(f"No saved model for problem_id='{problem_id}'")

    if require_stage is not None:
        stage = registry.models.metadata(problem_id, v).get("stage")
        if stage != require_stage:
            raise PermissionError(
                f"Model '{problem_id}' v{v} is at stage='{stage}', not '{require_stage}' -- "
                f"promote it first via registry_tasks.publish_model_task(..., stage='{require_stage}', confirm=True)."
            )

    model = registry.models.load(problem_id, v, model_cls=model_cls, **init_kwargs)
    return DigitalTwin(model, field_names, coord_names, config=config or DigitalTwinConfig())
