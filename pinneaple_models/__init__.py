from .base import BaseModel, ModelOutput
from .registry import ModelRegistry, ModelSpec
from .instantiate import InstantiateReport
from .catalog import ModelCatalog
from .adapters.base import select_adapter

# Group B architectures (standalone building blocks)
from .siren import SIREN, SineLayer
from .modified_mlp import ModifiedMLP, FourierFeatureEmbedding
from .hash_grid import HashGridMLP, HashGridEncoding
from .afno import AFNO, AFNOLayer
from .group_b_registry import GroupBCatalog

# MeshGraphNet — canonical implementation lives in graphnn
from .graphnn.mesh_graph_net import MeshGraphNet

# Populate ModelRegistry with all families on import
from .register_all import register_all as _register_all
_register_all()

__all__ = [
    # Base
    "BaseModel",
    "ModelOutput",
    # Registry
    "ModelRegistry",
    "ModelSpec",
    "InstantiateReport",
    "ModelCatalog",
    "select_adapter",
    # Group B architectures
    "SIREN",
    "SineLayer",
    "ModifiedMLP",
    "FourierFeatureEmbedding",
    "HashGridMLP",
    "HashGridEncoding",
    "MeshGraphNet",
    "AFNO",
    "AFNOLayer",
    "GroupBCatalog",
]
