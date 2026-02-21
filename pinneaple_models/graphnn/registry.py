from __future__ import annotations
"""Registry and catalog for graph neural network model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import GraphModelBase
from .gnn import GraphNeuralNetwork
from .equivariant_gnn import EquivariantGNN
from .gnn_ode import GraphNeuralODE
from .spatiotemporal_gnn import SpatiotemporalGNN
from .graphcast import GraphCast


_REGISTRY: Dict[str, Type[GraphModelBase]] = {
    "gnn": GraphNeuralNetwork,
    "graph_neural_network": GraphNeuralNetwork,

    "equivariant_gnn": EquivariantGNN,
    "egnn": EquivariantGNN,

    "graph_neural_ode": GraphNeuralODE,
    "gnn_ode": GraphNeuralODE,

    "spatiotemporal_gnn": SpatiotemporalGNN,
    "stgnn": SpatiotemporalGNN,

    "graphcast": GraphCast,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="graphnn")

@dataclass
class GraphCatalog:
    registry: Dict[str, Type[GraphModelBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[GraphModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown graph model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> GraphModelBase:
        return self.get(name)(**kwargs)
