"""pinneaple_models — compatibility shim.

Re-exports from pinneaple_neural.architectures so that legacy code importing
``from pinneaple_models.*`` continues to work.

All new code should import directly from ``pinneaple_neural.architectures``.
"""
from pinneaple_neural.architectures import (
    BaseModel,
    ModelRegistry,
    ModelCatalog,
)

# Top-level convenience re-exports
from pinneaple_neural.architectures.pinns.vanilla   import VanillaPINN
from pinneaple_neural.architectures.siren           import SIREN
from pinneaple_neural.architectures.modified_mlp    import ModifiedMLP
from pinneaple_neural.architectures.neural_operators.fno      import FourierNeuralOperator, FNO2d
from pinneaple_neural.architectures.neural_operators.deeponet import DeepONet
from pinneaple_neural.architectures.graphnn.mesh_graph_net    import MeshGraphNet
from pinneaple_neural.architectures.graphnn.base              import GraphBatch, GraphOutput

# Sub-namespace aliases used by legacy examples
class _PINNsNamespace:
    VanillaPINN = VanillaPINN
    SIREN       = SIREN
    ModifiedMLP = ModifiedMLP
    try:
        from pinneaple_neural.architectures.pinns.xtfc import XTFC
    except Exception:
        XTFC = None

class _NeuralOperatorsNamespace:
    FourierNeuralOperator = FourierNeuralOperator
    FNO2d     = FNO2d
    DeepONet  = DeepONet

class _GraphNNNamespace:
    MeshGraphNet = MeshGraphNet
    GraphBatch   = GraphBatch

pinns            = _PINNsNamespace()
neural_operators = _NeuralOperatorsNamespace()
graphnn          = _GraphNNNamespace()

try:
    from pinneaple_neural.architectures._catalog import register_all
except Exception:
    def register_all(): pass  # type: ignore

def registry():
    return ModelRegistry

catalog = ModelCatalog

__all__ = [
    "BaseModel", "ModelRegistry", "ModelCatalog",
    "VanillaPINN", "SIREN", "ModifiedMLP",
    "FourierNeuralOperator", "FNO2d", "DeepONet",
    "MeshGraphNet", "GraphBatch", "GraphOutput",
    "pinns", "neural_operators", "graphnn",
    "register_all", "registry", "catalog",
]
