from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleSpec:
    """
    Optional nondimensionalization / scaling parameters.
    Keep minimal; advanced scaling can be built later.

    L: characteristic length
    U: characteristic velocity (if any)
    alpha: thermal diffusivity (if any)
    """
    L: float = 1.0
    U: float = 1.0
    alpha: float = 1.0