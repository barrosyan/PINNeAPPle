from __future__ import annotations
"""Registry and catalog for neural operator model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import NeuralOperatorBase
from .deeponet import DeepONet
from .fno import FourierNeuralOperator
from .gno import GalerkinNeuralOperator
from .ms_deeponet import MultiScaleDeepONet
from .pino import PhysicsInformedNeuralOperator
from .uno import UniversalNeuralOperator


_REGISTRY: Dict[str, Type[NeuralOperatorBase]] = {
    "deeponet": DeepONet,
    "multiscale_deeponet": MultiScaleDeepONet,

    "fno": FourierNeuralOperator,
    "fourier_neural_operator": FourierNeuralOperator,

    "gno": GalerkinNeuralOperator,
    "galerkin_neural_operator": GalerkinNeuralOperator,

    "pino": PhysicsInformedNeuralOperator,
    "physics_informed_neural_operator": PhysicsInformedNeuralOperator,

    "uno": UniversalNeuralOperator,
    "universal_operator_network": UniversalNeuralOperator,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="neural_operators")

@dataclass
class NeuralOperatorCatalog:
    registry: Dict[str, Type[NeuralOperatorBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[NeuralOperatorBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown neural operator '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> NeuralOperatorBase:
        return self.get(name)(**kwargs)
