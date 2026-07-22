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

_SEQUENCE_MODELS = (EchoStateNetwork, ESNRC, KoopmanOperator)  # forward(x: (B,T,in_dim), ...)


def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        # ELM/RBF/HybridRBF are row-wise (each point independent, see
        # elm.py:78/rbf.py:233 forward(x: (N,in_dim))) — genuinely pointwise.
        # ESN/ESNRC/Koopman require x: (B,T,in_dim) and roll out over T
        # (esn.py:296, koopman.py:139) — real sequence models.
        if cls in _SEQUENCE_MODELS:
            return {"input_kind": "sequence", "expects": ["x_past"], "predicts": ["u"]}
        return {"input_kind": "pointwise_coords", "expects": ["x"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="reservoir_computing", capabilities_getter=capabilities)

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
