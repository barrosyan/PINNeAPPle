"""pinneapple_orchestration.tasks.registry_tasks — the task behind
flow_04_registry_publish, the terminal "promote to a real stage" step.

Requires an explicit ``confirm=True``, the same human-confirmation gate a
sibling internal project's own library-publishing flow used for its
terminal step — a stage promotion (especially to ``"production"``) is
exactly the kind of action that should never happen as a side effect of
an otherwise-automated pipeline running to completion.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pinneapple_registry import ArtifactRegistry

from .._prefect_compat import task


@task
def publish_model_task(
    registry: ArtifactRegistry,
    problem_id: str,
    model: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    stage: str = "staging",
    confirm: bool = False,
) -> str:
    """Saves ``model`` into ``registry.models`` and (only if
    ``confirm=True``) promotes it to ``stage``. Raises ``PermissionError``
    if ``confirm`` is not explicitly ``True`` and ``stage`` is anything
    other than ``"development"`` — the default, un-promoted stage a plain
    ``save()`` already lands on."""
    version = registry.models.save(problem_id, model, metadata=metadata, stage="development")
    if stage != "development":
        if not confirm:
            raise PermissionError(
                f"Refusing to promote '{problem_id}' v{version} to stage='{stage}' without confirm=True "
                "-- stage promotion is a deliberate, human-confirmed action, not an automatic pipeline step."
            )
        registry.models.promote(problem_id, version, stage)
    return version
