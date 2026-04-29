from __future__ import annotations
"""Registry for Group B neural architectures.

Covers:
  - SIREN (Sinusoidal Representation Networks)
  - ModifiedMLP (Fourier features + highway U/V gating)
  - HashGridMLP (Instant-NGP style multi-resolution hash encoding)
  - MeshGraphNet (GNN for unstructured FEM meshes)
  - AFNO (Adaptive Fourier Neural Operator)
"""

from dataclasses import dataclass
from typing import Dict, Type

import torch.nn as nn

from .siren import SIREN
from .modified_mlp import ModifiedMLP
from .hash_grid import HashGridMLP
from .mesh_graph_net import MeshGraphNet
from .afno import AFNO


_REGISTRY: Dict[str, Type[nn.Module]] = {
    # --- SIREN -----------------------------------------------------------
    "siren": SIREN,
    "sinusoidal_representation_network": SIREN,

    # --- Modified MLP ----------------------------------------------------
    "modified_mlp": ModifiedMLP,
    "fourier_highway_mlp": ModifiedMLP,

    # --- Hash Grid MLP ---------------------------------------------------
    "hash_grid_mlp": HashGridMLP,
    "instant_ngp_mlp": HashGridMLP,

    # --- MeshGraphNet ----------------------------------------------------
    "mesh_graph_net": MeshGraphNet,
    "mgn": MeshGraphNet,

    # --- AFNO ------------------------------------------------------------
    "afno": AFNO,
    "adaptive_fourier_neural_operator": AFNO,
}

# Family → input_kind / capability annotations
_CAPABILITIES: Dict[str, Dict] = {
    "siren":               {"input_kind": "pointwise_coords", "supports_physics_loss": True},
    "modified_mlp":        {"input_kind": "pointwise_coords", "supports_physics_loss": True},
    "hash_grid_mlp":       {"input_kind": "pointwise_coords", "supports_physics_loss": False},
    "mesh_graph_net":      {"input_kind": "graph",            "supports_physics_loss": False},
    "afno":                {"input_kind": "grid",             "supports_physics_loss": False},
}


def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry

    def caps_getter(name: str, cls: Type[nn.Module]) -> Dict:
        # Look up by canonical name (first key that maps to this class)
        for canonical, model_cls in _REGISTRY.items():
            if model_cls is cls and canonical in _CAPABILITIES:
                return _CAPABILITIES[canonical]
        return {}

    register_family_registry(
        _REGISTRY,
        family="group_b",
        capabilities_getter=caps_getter,
    )


@dataclass
class GroupBCatalog:
    registry: Dict[str, Type[nn.Module]] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[nn.Module]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown Group-B model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> nn.Module:
        return self.get(name)(**kwargs)
