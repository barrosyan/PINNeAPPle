"""Solver catalog registry for FFT, Hilbert-Huang, FEM, FVM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from .base import SolverBase
from .fft import FFTSolver
from .hilbert_huang import HilbertHuangSolver
from .fem import FEMSolver
from .fvm import FVMSolver


_REGISTRY: Dict[str, Type[SolverBase]] = {
    "fft": FFTSolver,
    "fft_solver": FFTSolver,

    "hilbert_huang": HilbertHuangSolver,
    "hht": HilbertHuangSolver,

    "fem": FEMSolver,
    "finite_element_method": FEMSolver,

    "fvm": FVMSolver,
    "finite_volume_method": FVMSolver,
}


@dataclass
class SolverCatalog:
    """Registry of solver implementations (FFT, Hilbert-Huang, FEM, FVM)."""

    registry: Dict[str, Type[SolverBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[SolverBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown solver '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> SolverBase:
        return self.get(name)(**kwargs)
