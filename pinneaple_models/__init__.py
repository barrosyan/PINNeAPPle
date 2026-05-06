from .registry import ModelRegistry, ModelSpec

# Group B architectures
from .siren import SIREN, SineLayer
from .modified_mlp import ModifiedMLP, FourierFeatureEmbedding
from .hash_grid import HashGridMLP, HashGridEncoding
from .mesh_graph_net import MeshGraphNet
from .afno import AFNO, AFNOLayer
from .group_b_registry import GroupBCatalog

# Populate ModelRegistry with all families on import
from .register_all import register_all as _register_all
_register_all()

__all__ = [
    "ModelRegistry",
    "ModelSpec",
    # Group B
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
    # Registration
    "register_all",
]
