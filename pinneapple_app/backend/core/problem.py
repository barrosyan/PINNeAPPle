"""Problem definition layer for pinneapple_app.

Two paths:
- Preset  : load from pinneapple_environment registry by name
- Custom  : user supplies SymPy/string equations + BCs + ICs
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Domain family constants (used for model recommendations) ────────────────
FAMILY_FLUID       = "fluid"
FAMILY_THERMAL     = "thermal"
FAMILY_STRUCTURAL  = "structural"
FAMILY_WAVE        = "wave"
FAMILY_DIFFUSION   = "diffusion"
FAMILY_FINANCE     = "finance"
FAMILY_BIOLOGICAL  = "biological"
FAMILY_GENERIC     = "generic"


@dataclass
class BoundaryConditionSpec:
    """Human-readable BC for custom problems."""
    kind: str          # "dirichlet" | "neumann" | "robin" | "periodic"
    location: str      # e.g. "x=0", "left", "top"
    value: Any         # float, expression string, or callable
    field: str = "u"


@dataclass
class InitialConditionSpec:
    """Human-readable IC for custom problems."""
    expression: Any    # float, string expression, or callable
    field: str = "u"


@dataclass
class EquationSpec:
    """A single PDE residual equation (string or SymPy expression)."""
    expression: Any    # SymPy Expr or string like "u_t + u*u_x - nu*u_xx"
    label: str = "pde"
    field: str = "u"


@dataclass
class ProblemDefinition:
    """Unified problem container used throughout pinneapple_app."""
    kind: str                          # "preset" | "custom"
    name: str                          # display name
    spec: Any                          # pinneapple_environment.ProblemSpec
    dim: int                           # spatial dimension
    domain_bounds: Dict[str, Tuple[float, float]]
    pde_family: str = FAMILY_GENERIC   # fluid / thermal / structural / …
    is_time_dependent: bool = False
    preset_name: Optional[str] = None  # original registry key (preset only)
    equations: List[EquationSpec] = field(default_factory=list)   # custom only
    bcs: List[BoundaryConditionSpec] = field(default_factory=list) # custom only
    ics: List[InitialConditionSpec] = field(default_factory=list)  # custom only
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ────────────────────────────────────────────────────────

    @property
    def coord_names(self) -> Tuple[str, ...]:
        if self.spec is not None and hasattr(self.spec, "coords"):
            return tuple(self.spec.coords)
        return tuple(self.domain_bounds.keys())

    @property
    def field_names(self) -> Tuple[str, ...]:
        if self.spec is not None and hasattr(self.spec, "fields"):
            return tuple(self.spec.fields)
        unique = list({eq.field for eq in self.equations} or {"u"})
        return tuple(sorted(unique))

    def __str__(self) -> str:
        return (f"ProblemDefinition(name={self.name!r}, kind={self.kind!r}, "
                f"family={self.pde_family!r}, dim={self.dim})")


# ── Loaders ────────────────────────────────────────────────────────────────

def load_preset(preset_name: str) -> ProblemDefinition:
    """Load a ProblemDefinition from the environment preset registry."""
    from pinneapple_physics.pde_environment import identify_pde
    from .problem_registry import get_problem_meta

    meta = get_problem_meta(preset_name)

    try:
        from pinneapple_physics.pde_environment import get_preset
        spec = get_preset(preset_name)
    except Exception:
        spec = None   # preset factory failed — continue without spec

    bounds = (getattr(spec, "domain_bounds", {}) or {}) if spec is not None else {}
    # fallback bounds from metadata tags
    if not bounds:
        dim_hint = meta.get("dim", 2)
        bounds = {ax: (0.0, 1.0) for ax in ["x", "y", "z", "t"][:dim_hint]}
    dim = getattr(spec, "dim", len(bounds)) if spec is not None else len(bounds)

    pde_ids = identify_pde(preset_name.replace("_", " "))
    pde_family = meta.get("family", _family_from_identify(pde_ids))

    display_name = (spec.name if spec is not None and hasattr(spec, "name") and spec.name
                    else preset_name)

    return ProblemDefinition(
        kind="preset",
        name=display_name,
        spec=spec,
        dim=dim,
        domain_bounds=bounds,
        pde_family=pde_family,
        is_time_dependent=meta.get("time_dependent", _has_time(spec)),
        preset_name=preset_name,
        meta=meta,
    )


def define_custom(
    name: str,
    equations: List[EquationSpec],
    bcs: List[BoundaryConditionSpec],
    ics: Optional[List[InitialConditionSpec]] = None,
    domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    dim: int = 2,
    pde_family: str = FAMILY_GENERIC,
    is_time_dependent: bool = False,
) -> ProblemDefinition:
    """Build a ProblemDefinition from user-supplied equations, BCs, and ICs.

    This path does not require a ProblemSpec; the equations are stored as
    EquationSpec objects and compiled into physics losses at experiment time.
    """
    bounds = domain_bounds or {
        f"x{i}": (0.0, 1.0) for i in range(dim)
    }
    return ProblemDefinition(
        kind="custom",
        name=name,
        spec=None,
        dim=dim,
        domain_bounds=bounds,
        pde_family=pde_family,
        is_time_dependent=is_time_dependent,
        equations=equations,
        bcs=bcs,
        ics=ics or [],
    )


def _has_time(spec) -> bool:
    coords = getattr(spec, "coords", []) or []
    return "t" in coords


def _family_from_identify(pde_ids: list) -> str:
    if not pde_ids:
        return FAMILY_GENERIC
    top = pde_ids[0][0] if pde_ids else ""
    _MAP = {
        "navier_stokes": FAMILY_FLUID,
        "stokes":        FAMILY_FLUID,
        "darcy":         FAMILY_FLUID,
        "heat":          FAMILY_THERMAL,
        "diffusion":     FAMILY_DIFFUSION,
        "wave":          FAMILY_WAVE,
        "helmholtz":     FAMILY_WAVE,
        "elasticity":    FAMILY_STRUCTURAL,
        "burgers":       FAMILY_FLUID,
        "poisson":       FAMILY_DIFFUSION,
        "laplace":       FAMILY_DIFFUSION,
        "black_scholes": FAMILY_FINANCE,
        "sir":           FAMILY_BIOLOGICAL,
    }
    for key, fam in _MAP.items():
        if key in top:
            return fam
    return FAMILY_GENERIC
