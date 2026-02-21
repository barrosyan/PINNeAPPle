from __future__ import annotations
"""Registry and catalog for reduced-order model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import ROMBase
from .pod import POD
from .dmd import DynamicModeDecomposition
from .havok import HAVOK
from .opinf import OperatorInference
from .rom_hybrid import ROMHybrid
from .deep_uq_rom import DeepUQROM


_REGISTRY: Dict[str, Type[ROMBase]] = {
    "pod": POD,
    "proper_orthogonal_decomposition": POD,

    "dmd": DynamicModeDecomposition,
    "dynamic_mode_decomposition": DynamicModeDecomposition,

    "havok": HAVOK,

    "operator_inference": OperatorInference,
    "opinf": OperatorInference,

    "rom_hybrid": ROMHybrid,

    "deep_uq_rom": DeepUQROM,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="rom")

@dataclass
class ROMCatalog:
    registry: Dict[str, Type[ROMBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[ROMBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown ROM model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> ROMBase:
        return self.get(name)(**kwargs)
