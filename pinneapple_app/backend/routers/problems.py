"""Problems API router."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/problems", tags=["problems"])


class PresetListItem(BaseModel):
    name: str
    family: str
    description: str
    tags: List[str]
    time_dependent: bool
    recommended_models: List[str]
    recommended_solver: str


class CustomProblemRequest(BaseModel):
    name: str
    equations: List[str]           # SymPy-compatible expression strings
    boundary_conditions: List[Dict[str, Any]]
    initial_conditions: List[Dict[str, Any]] = []
    domain_bounds: Dict[str, List[float]]     # {"x": [0, 1], "y": [0, 1]}
    dim: int = 2
    pde_family: str = "generic"
    is_time_dependent: bool = False


@router.get("", response_model=List[PresetListItem])
def list_problems():
    """Return all available preset problems with metadata."""
    from ..core.problem_registry import all_problems, get_problem_meta
    items = []
    for name in all_problems():
        meta = get_problem_meta(name)
        items.append(PresetListItem(
            name=name,
            family=meta["family"],
            description=meta["description"],
            tags=meta["tags"],
            time_dependent=meta["time_dependent"],
            recommended_models=meta["recommended_models"],
            recommended_solver=meta["recommended_solver"],
        ))
    return items


@router.get("/{name}", response_model=PresetListItem)
def get_problem(name: str):
    """Return metadata for a single preset problem."""
    from ..core.problem_registry import all_problems, get_problem_meta
    if name not in all_problems():
        raise HTTPException(status_code=404, detail=f"Problem '{name}' not found.")
    meta = get_problem_meta(name)
    return PresetListItem(
        name=name,
        family=meta["family"],
        description=meta["description"],
        tags=meta["tags"],
        time_dependent=meta["time_dependent"],
        recommended_models=meta["recommended_models"],
        recommended_solver=meta["recommended_solver"],
    )


@router.post("/custom/validate")
def validate_custom_problem(req: CustomProblemRequest):
    """Validate a custom problem definition without running it."""
    errors = []
    if not req.equations:
        errors.append("At least one equation is required.")
    if not req.boundary_conditions:
        errors.append("At least one boundary condition is required.")
    if req.dim < 1 or req.dim > 4:
        errors.append("dim must be between 1 and 4.")
    if len(req.domain_bounds) == 0:
        errors.append("domain_bounds must not be empty.")
    for coord, bounds in req.domain_bounds.items():
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            errors.append(f"Invalid bounds for {coord!r}: {bounds}")
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return {"valid": True, "message": "Custom problem definition is valid."}
