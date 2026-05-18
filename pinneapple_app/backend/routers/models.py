"""Models API router."""
from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelListItem(BaseModel):
    name: str
    family: str
    description: str
    supports_physics_loss: bool
    tags: List[str]
    recommended_for: List[str]


class MetricItem(BaseModel):
    key: str
    label: str


@router.get("", response_model=List[ModelListItem])
def list_models(family: Optional[str] = None):
    """Return all available models, optionally filtered by family."""
    from ..core.model_catalog import list_models as _list
    return [
        ModelListItem(
            name=e.name,
            family=e.family,
            description=e.description,
            supports_physics_loss=e.supports_physics_loss,
            tags=e.tags,
            recommended_for=e.recommended_for,
        )
        for e in _list(family=family)
    ]


@router.get("/families")
def list_families():
    """Return all model families."""
    try:
        from pinneapple_neural.architectures import ModelRegistry
        return {"families": ModelRegistry.families()}
    except Exception:
        return {"families": []}


@router.get("/recommend/{problem_family}")
def recommend_models(problem_family: str, n: int = 5):
    """Return top-N recommended model names for a problem family."""
    from ..core.model_catalog import recommend_for_problem
    return {"recommendations": recommend_for_problem(problem_family, n)}


@router.get("/metrics")
def list_metrics():
    """Return all available evaluation metrics."""
    from ..core.model_catalog import AVAILABLE_METRICS, DEFAULT_METRICS
    return {
        "available": [{"key": k, "label": v} for k, v in AVAILABLE_METRICS.items()],
        "defaults":  DEFAULT_METRICS,
    }


@router.get("/{name}")
def get_model(name: str):
    """Return metadata for a single model."""
    from ..core.model_catalog import get_model_info
    entry = get_model_info(name)
    if entry is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")
    return ModelListItem(
        name=entry.name,
        family=entry.family,
        description=entry.description,
        supports_physics_loss=entry.supports_physics_loss,
        tags=entry.tags,
        recommended_for=entry.recommended_for,
    )
