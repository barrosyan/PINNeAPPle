"""Solver registry.

Pinneaple uses "solvers" in two distinct ways:
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

def register_all() -> None:
    """Import solver modules so their @register decorators execute."""

    from . import fft as _fft  # noqa: F401
    from . import hilbert_huang as _hht  # noqa: F401
    from . import eemd as _eemd  # noqa: F401
    from . import ceemdan as _ceemdan  # noqa: F401
    from . import vmd as _vmd  # noqa: F401
    from . import wavelet as _wav  # noqa: F401
    from . import sst as _sst  # noqa: F401
    from . import ssa as _ssa  # noqa: F401
    from . import stl as _stl  # noqa: F401

    from . import fdm as _fdm  # noqa: F401
    from . import fem as _fem  # noqa: F401
    from . import fvm as _fvm  # noqa: F401
    from . import lbm as _lbm  # noqa: F401
    from . import spectral as _spectral  # noqa: F401

    from . import sph as _sph  # noqa: F401
    from . import isph as _isph  # noqa: F401
    from . import dfsph as _dfsph  # noqa: F401

    from . import meshfree as _meshfree  # noqa: F401


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
