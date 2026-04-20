"""pinneaple_capabilities — PDE family knowledge base.

Maps natural-language problem descriptions and structured answers to known
PDE families, equation kinds, and parameter sets used by ProblemSpec.

Used by pinneaple_problemdesign.DesignAgent to identify PDE kind strings
from user-provided physics descriptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# PDE family descriptor
# ---------------------------------------------------------------------------

@dataclass
class PDEFamily:
    """Descriptor for a known PDE family.

    Attributes
    ----------
    kind : str
        Canonical kind string accepted by compile_problem and ProblemSpec.
    aliases : list of str
        Alternative names / common descriptions that map to this family.
    description : str
        Human-readable description.
    default_params : dict
        Default parameter values (can be overridden by user).
    coords_hint : list of str
        Suggested coordinate names (e.g. ["x", "t"] for 1D transient).
    fields_hint : list of str
        Suggested field names.
    tags : list of str
        Domain tags for filtering (e.g. "cfd", "heat", "mechanics").
    """
    kind: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    coords_hint: List[str] = field(default_factory=list)
    fields_hint: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, PDEFamily] = {}


def _reg(family: PDEFamily) -> PDEFamily:
    _REGISTRY[family.kind] = family
    for alias in family.aliases:
        _REGISTRY[alias.lower()] = family
    return family


# Heat / diffusion
_reg(PDEFamily(
    kind="heat_equation",
    aliases=["heat", "heat_1d", "diffusion", "heat equation", "thermal diffusion", "parabolic"],
    description="Transient heat (diffusion) equation: u_t = alpha * nabla^2 u",
    default_params={"alpha": 0.01},
    coords_hint=["x", "t"],
    fields_hint=["u"],
    tags=["heat", "diffusion"],
))

# Burgers
_reg(PDEFamily(
    kind="burgers",
    aliases=["burgers_1d", "viscous burgers", "burgers equation", "inviscid burgers"],
    description="Burgers equation: u_t + u * u_x = nu * u_xx",
    default_params={"nu": 0.01},
    coords_hint=["x", "t"],
    fields_hint=["u"],
    tags=["cfd", "nonlinear"],
))

# Poisson
_reg(PDEFamily(
    kind="poisson",
    aliases=["poisson_2d", "poisson equation", "electrostatics"],
    description="Poisson equation: -nabla^2 u = f",
    default_params={"f": 1.0},
    coords_hint=["x", "y"],
    fields_hint=["u"],
    tags=["elliptic", "heat"],
))

# Laplace
_reg(PDEFamily(
    kind="laplace",
    aliases=["laplace_2d", "laplace equation", "harmonic"],
    description="Laplace equation: nabla^2 u = 0",
    default_params={},
    coords_hint=["x", "y"],
    fields_hint=["u"],
    tags=["elliptic"],
))

# Navier-Stokes
_reg(PDEFamily(
    kind="navier_stokes",
    aliases=[
        "ns", "navier stokes", "incompressible flow", "cfd", "fluid dynamics",
        "navier_stokes_2d", "ns_incompressible_2d",
    ],
    description="Incompressible Navier-Stokes: rho(u_t + u*grad(u)) = -grad(p) + mu*nabla^2 u, div(u)=0",
    default_params={"Re": 100.0, "rho": 1.0, "mu": 0.01},
    coords_hint=["x", "y", "t"],
    fields_hint=["u", "v", "p"],
    tags=["cfd", "fluid"],
))

# Wave equation
_reg(PDEFamily(
    kind="wave",
    aliases=["wave equation", "acoustic", "second order hyperbolic"],
    description="Wave equation: u_tt = c^2 * nabla^2 u",
    default_params={"c": 1.0},
    coords_hint=["x", "t"],
    fields_hint=["u"],
    tags=["wave", "hyperbolic"],
))

# Helmholtz
_reg(PDEFamily(
    kind="helmholtz",
    aliases=["helmholtz equation", "frequency domain wave"],
    description="Helmholtz equation: nabla^2 u + k^2 u = f",
    default_params={"k": 1.0},
    coords_hint=["x", "y"],
    fields_hint=["u"],
    tags=["wave", "elliptic"],
))

# Reaction-diffusion
_reg(PDEFamily(
    kind="reaction_diffusion",
    aliases=["reaction diffusion", "fisher", "gray scott", "turing"],
    description="Reaction-diffusion: u_t = D * nabla^2 u + R(u)",
    default_params={"D": 0.001, "k": 0.1},
    coords_hint=["x", "y", "t"],
    fields_hint=["u"],
    tags=["diffusion", "biology"],
))

# Darcy flow
_reg(PDEFamily(
    kind="darcy",
    aliases=["darcy flow", "porous media", "darcy equation"],
    description="Darcy flow: -div(K * grad(p)) = f",
    default_params={"K": 1.0},
    coords_hint=["x", "y"],
    fields_hint=["p"],
    tags=["cfd", "porous"],
))

# Linear elasticity
_reg(PDEFamily(
    kind="linear_elasticity",
    aliases=["elasticity", "solid mechanics", "stress strain", "structural"],
    description="Linear elasticity: div(sigma) + b = 0",
    default_params={"E": 200e9, "nu": 0.3},
    coords_hint=["x", "y"],
    fields_hint=["u", "v"],
    tags=["mechanics", "structural"],
))

# Black-Scholes
_reg(PDEFamily(
    kind="black_scholes",
    aliases=["black scholes", "option pricing", "finance pde"],
    description="Black-Scholes PDE for option pricing",
    default_params={"r": 0.05, "sigma": 0.2},
    coords_hint=["S", "t"],
    fields_hint=["V"],
    tags=["finance"],
))

# SIR
_reg(PDEFamily(
    kind="sir",
    aliases=["sir model", "epidemiology", "epidemic"],
    description="SIR epidemiological ODE system",
    default_params={"beta": 0.3, "gamma": 0.1},
    coords_hint=["t"],
    fields_hint=["S", "I", "R"],
    tags=["biology", "ode"],
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_pde_families() -> List[str]:
    """Return sorted list of canonical PDE kind strings."""
    return sorted({f.kind for f in _REGISTRY.values()})


def get_pde_family(kind_or_alias: str) -> Optional[PDEFamily]:
    """Look up a PDEFamily by kind string or alias (case-insensitive)."""
    key = kind_or_alias.strip().lower()
    return _REGISTRY.get(key) or _REGISTRY.get(kind_or_alias)


def identify_pde(description: str) -> List[Tuple[str, float]]:
    """Score known PDE families against a free-text description.

    Returns a ranked list of (kind, score) tuples, highest score first.
    Score is a simple keyword overlap count — good enough for guided elicitation.
    """
    desc_lower = description.lower()
    scores: Dict[str, float] = {}
    for kind, family in _REGISTRY.items():
        score = 0.0
        # Check alias matches
        for alias in [family.kind] + family.aliases:
            if alias.lower() in desc_lower:
                score += 2.0
        # Check tag matches
        for tag in family.tags:
            if tag in desc_lower:
                score += 1.0
        # Check description words
        for word in family.description.lower().split():
            if len(word) > 4 and word in desc_lower:
                score += 0.5
        if score > 0:
            scores[family.kind] = max(scores.get(family.kind, 0.0), score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def suggest_problem_spec(description: str) -> Dict[str, Any]:
    """Given a free-text description, suggest ProblemSpec constructor arguments.

    Returns a dict with:
      - 'kind': canonical PDE kind string
      - 'fields': suggested field names
      - 'coords': suggested coordinate names
      - 'default_params': default parameter dict
      - 'tags': domain tags
      - 'confidence': match score
    """
    ranked = identify_pde(description)
    if not ranked:
        return {
            "kind": "custom",
            "fields": ["u"],
            "coords": ["x", "t"],
            "default_params": {},
            "tags": [],
            "confidence": 0.0,
        }

    best_kind, best_score = ranked[0]
    family = get_pde_family(best_kind)
    assert family is not None

    return {
        "kind": family.kind,
        "fields": list(family.fields_hint) or ["u"],
        "coords": list(family.coords_hint) or ["x"],
        "default_params": dict(family.default_params),
        "tags": list(family.tags),
        "confidence": best_score,
    }


__all__ = [
    "PDEFamily",
    "list_pde_families",
    "get_pde_family",
    "identify_pde",
    "suggest_problem_spec",
]
