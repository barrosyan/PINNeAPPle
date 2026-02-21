from __future__ import annotations
"""Registry and catalog for physics-aware model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import PhysicsAwareBase
from .pann import PhysicsAwareNeuralNetwork
from .spn import StructurePreservingNetwork


_REGISTRY: Dict[str, Type[PhysicsAwareBase]] = {
    "physics_aware_neural_network": PhysicsAwareNeuralNetwork,
    "pann": PhysicsAwareNeuralNetwork,

    "structure_preserving_network": StructurePreservingNetwork,
    "spn": StructurePreservingNetwork,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="physics_aware")

@dataclass
class PhysicsAwareCatalog:
    registry: Dict[str, Type[PhysicsAwareBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[PhysicsAwareBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown physics_aware model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> PhysicsAwareBase:
        return self.get(name)(**kwargs)
