"""Solver registry.

Pinneapple uses "solvers" in two distinct ways:
  1) Numerical PDE solvers (FDM/FEM/FVM/LBM/Spectral)
  2) Signal/feature solvers for time-series preprocessing (FFT/HHT/Wavelet/SSA/...)
  3) Particle methods (SPH variants)

This registry gives the Arena and training pipelines a unified catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from .base import SolverBase


@dataclass
class SolverSpec:
    name: str
    family: str
    cls: Type[SolverBase]
    description: str = ""
    tags: List[str] | None = None


class SolverRegistry:
    _REG: Dict[str, SolverSpec] = {}

    @classmethod
    def register(
        cls,
        *,
        name: str,
        family: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ):
        key = str(name).lower().strip()
        if not key:
            raise ValueError("Solver name must be non-empty")

        def deco(solver_cls: Type[SolverBase]):
            if key in cls._REG and cls._REG[key].cls is not solver_cls:
                raise KeyError(f"Solver '{key}' already registered")
            cls._REG[key] = SolverSpec(name=key, family=str(family), cls=solver_cls, description=description, tags=tags or [])
            return solver_cls

        return deco

    @classmethod
    def spec(cls, name: str) -> SolverSpec:
        key = str(name).lower().strip()
        if key not in cls._REG:
            raise KeyError(f"Unknown solver '{name}'. Available: {cls.list()}")
        return cls._REG[key]

    @classmethod
    def list(cls, family: Optional[str] = None) -> List[str]:
        if family is None:
            return sorted(cls._REG.keys())
        fam = str(family)
        return sorted([k for k, v in cls._REG.items() if v.family == fam])

    @classmethod
    def build(cls, name: str, **kwargs) -> SolverBase:
        spec = cls.spec(name)
        return spec.cls(**kwargs)


# --------- register built-ins
# Keep imports at bottom to avoid import cycles.

_ALL_SOLVER_MODULES = [
    "fft", "hilbert_huang", "eemd", "ceemdan", "vmd", "wavelet", "sst",
    "ssa", "stl",
    "fdm", "fem", "fvm", "lbm", "spectral",
    "sph", "isph", "dfsph",
    "meshfree",
    "xtfc_ivp", "eddy_current_fdm", "immersed_boundary_fdm", "beam_bvp_fdm",
    "elasticity3d_fdm", "nonlinear_beam_fem",
]


def register_all() -> Dict[str, str]:
    """Import solver modules so their @register decorators execute.

    Each module is imported independently: one solver needing an
    uninstalled optional dependency (e.g. `wavelet` needs `pywt`) used to
    raise straight out of this function and silently skip registering
    every solver listed *after* it too (a plain top-to-bottom list of
    `from . import x` statements aborts on the first failure) -- found via
    `SolverRegistry.list()` returning only 8 of the ~26 solvers that
    exist in this package. Returns a dict of {module_name: error_message}
    for any that failed to import, so callers can inspect what's missing
    instead of that failure being invisible.
    """
    import importlib

    failures: Dict[str, str] = {}
    for mod_name in _ALL_SOLVER_MODULES:
        try:
            importlib.import_module(f".{mod_name}", __package__)
        except Exception as e:
            failures[mod_name] = str(e)
    return failures


@dataclass
class SolverCatalog:
    """
    Convenience wrapper (dict-like catalog) around SolverRegistry.

    Useful for code that prefers:
        cat = SolverCatalog(); solver = cat.build("eemd", ...)

    Auto-calls register_all() once so all solvers are available.
    """

    registry: SolverRegistry = None

    def __post_init__(self):
        register_all()
        self.registry = SolverRegistry()

    def list(self):
        return self.registry.list()

    def get(self, name: str):
        return self.registry.get(name)

    def build(self, name: str, **kwargs):
        return self.registry.build(name, **kwargs)
