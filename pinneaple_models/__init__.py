from .registry import ModelRegistry, ModelSpec

# Group B architectures
from .siren import SIREN, SineLayer
from .modified_mlp import ModifiedMLP, FourierFeatureEmbedding
from .hash_grid import HashGridMLP, HashGridEncoding
from .mesh_graph_net import MeshGraphNet
from .afno import AFNO, AFNOLayer
from .group_b_registry import GroupBCatalog

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
]
