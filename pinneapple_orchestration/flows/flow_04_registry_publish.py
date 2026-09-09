"""flow_04_registry_publish — the terminal, human-confirmed "promote to a
real stage" step."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pinneapple_registry import ArtifactRegistry

from .._prefect_compat import flow
from ..common.provenance import flow_run_manifest
from ..tasks import publish_model_task


@flow(name="flow_04_registry_publish")
def flow_registry_publish(
    registry: ArtifactRegistry,
    problem_id: str,
    model: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    stage: str = "staging",
    confirm: bool = False,
    manifest_dir: str = "data/manifests",
) -> Dict[str, Any]:
    with flow_run_manifest(
        "flow_04_registry_publish", {"problem_id": problem_id, "stage": stage, "confirm": confirm}, manifest_dir=manifest_dir
    ) as manifest:
        version = publish_model_task(registry, problem_id, model, metadata=metadata, stage=stage, confirm=confirm)
        manifest.log_task("publish", "ok", version=version, stage=stage)
        manifest.add_artifact("version", version)
        return {"problem_id": problem_id, "version": version, "stage": stage, "run_id": manifest.run_id}
