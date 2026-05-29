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
from .uno import UniversalUNO


_REGISTRY: Dict[str, Type[NeuralOperatorBase]] = {
    "deeponet": DeepONet,
    "multiscale_deeponet": MultiScaleDeepONet,

    "fno": FourierNeuralOperator,
    "fourier_neural_operator": FourierNeuralOperator,

    "gno": GalerkinNeuralOperator,
    "galerkin_neural_operator": GalerkinNeuralOperator,

    "pino": PhysicsInformedNeuralOperator,
    "physics_informed_neural_operator": PhysicsInformedNeuralOperator,

    "uno": UniversalUNO,
    "universal_operator_network": UniversalUNO,
}

# Noether (Emmi AI) models — registered lazily so emmiai-noether is optional
try:
    from .noether_bridge import NOETHER_REGISTRY
    _REGISTRY.update(NOETHER_REGISTRY)
except Exception:
    pass

def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry
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
