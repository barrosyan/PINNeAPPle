from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .typing import CoordNames, FieldNames
from .conditions import ConditionSpec
from .scales import ScaleSpec


@dataclass(frozen=True)
class PDETermSpec:
    """
    PDE descriptor used by the compiler.

    kind: name of PDE (e.g. laplace, poisson, burgers, navier_stokes_incompressible)
    fields: model output fields involved
    coords: coordinate names used by the PDE (e.g. ("x","y","t"))
    params: PDE parameters (e.g. nu, Re, kappa, alpha)
    meta: extra options (e.g. darcy mode)
    """
    kind: str
    fields: FieldNames
    coords: CoordNames
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemSpec:
    """
    Complete problem specification.

    conditions: boundary/initial/data constraints
    sample_defaults: recommended sampling counts
    """
    name: str
    dim: int
    coords: CoordNames
    fields: FieldNames
    pde: PDETermSpec
    conditions: Tuple[ConditionSpec, ...] = field(default_factory=tuple)

    sample_defaults: Dict[str, int] = field(default_factory=dict)
    scales: ScaleSpec = field(default_factory=ScaleSpec)
    field_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    references: Tuple[str, ...] = field(default_factory=tuple)