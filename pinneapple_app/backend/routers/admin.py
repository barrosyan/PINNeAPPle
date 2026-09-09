"""pinneapple_app.backend.routers.admin — internal curation/status view:
cross-references the problem catalog, the model registry
(``pinneapple_registry.ArtifactRegistry``), and dataset scenarios into one
dashboard-ready payload. Gated by ``core.admin_auth.require_admin`` (a
generic bearer token, see that module's docstring for why not an
email-domain allowlist).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..core.admin_auth import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _registry_root() -> str:
    return os.environ.get("PINNEAPPLE_REGISTRY_ROOT", "./pinneapple_registry_data")


class ModelSummaryItem(BaseModel):
    problem_id: str
    latest_model_version: Optional[str]
    stage: Optional[str] = None
    dataset_scenarios: List[str]


class AdminSummaryResponse(BaseModel):
    registry_root: str
    n_presets: int
    n_architectures: int
    models: List[ModelSummaryItem]


@router.get("/summary", response_model=AdminSummaryResponse)
def admin_summary() -> AdminSummaryResponse:
    from pinneapple_registry import ArtifactRegistry

    registry = ArtifactRegistry(_registry_root())
    summary = registry.summary()

    try:
        from pinneapple_physics.pde_environment import list_presets
        n_presets = len(list_presets())
    except Exception:
        n_presets = 0

    try:
        from pinneapple_neural.architectures import ModelRegistry
        n_architectures = len(ModelRegistry.list())
    except Exception:
        n_architectures = 0

    return AdminSummaryResponse(
        registry_root=summary["root"],
        n_presets=n_presets,
        n_architectures=n_architectures,
        models=[ModelSummaryItem(**row) for row in summary["problems"]],
    )


class PromoteRequest(BaseModel):
    problem_id: str
    version: str
    stage: str


@router.post("/models/promote")
def promote_model(req: PromoteRequest) -> Dict[str, Any]:
    """Promotes a stored model version to a new stage. Being behind
    ``require_admin`` already IS the confirmation gate here (unlike
    ``pinneapple_orchestration``'s flow-level ``confirm=True`` flag, which
    guards an unattended pipeline step instead of a human clicking a
    button in an admin UI)."""
    from pinneapple_registry import ArtifactRegistry

    registry = ArtifactRegistry(_registry_root())
    registry.models.promote(req.problem_id, req.version, req.stage)
    return {"problem_id": req.problem_id, "version": req.version, "stage": req.stage}
