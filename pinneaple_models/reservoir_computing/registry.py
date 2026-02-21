from __future__ import annotations
"""Registry and catalog for reservoir computing model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import RCBase
from .elm import ExtremeLearningMachine
from .rbf import RBFNetwork
from .hybrid_rbf import HybridRBFNetwork
from .esn import EchoStateNetwork
from .esn_rc import ESNRC
from .koopman import KoopmanOperator


_REGISTRY: Dict[str, Type[RCBase]] = {
    "elm": ExtremeLearningMachine,
    "extreme_learning_machine": ExtremeLearningMachine,

    "rbf": RBFNetwork,
    "rbf_network": RBFNetwork,

    "hybrid_rbf": HybridRBFNetwork,
    "hybrid_rbf_network": HybridRBFNetwork,

    "esn": EchoStateNetwork,
    "echo_state_network": EchoStateNetwork,

    "esn_rc": ESNRC,

    "koopman": KoopmanOperator,
    "koopman_operator": KoopmanOperator,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="reservoir_computing")

@dataclass
class ReservoirCatalog:
    registry: Dict[str, Type[RCBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[RCBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown reservoir model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> RCBase:
        return self.get(name)(**kwargs)
