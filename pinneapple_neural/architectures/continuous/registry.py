from __future__ import annotations
"""Registry and catalog for continuous-time model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import ContinuousModelBase
from .neural_ode import NeuralODE
from .ode_rnn import ODERNN
from .latent_ode import LatentODE
from .neural_cde import NeuralCDE
from .neural_sde import NeuralSDE
from .hamiltonian import HamiltonianNeuralNetwork
from .symplectic_ode import SymplecticODENet
from .symplectic_rnn import SymplecticRNN
from .bayesian_rnn import BayesianRNN
from .deep_state_space import DeepStateSpaceModel
from .neural_gp import NeuralGaussianProcess


_REGISTRY: Dict[str, Type[ContinuousModelBase]] = {
    "neural_ode": NeuralODE,
    "latent_ode": LatentODE,
    "ode_rnn": ODERNN,

    "neural_cde": NeuralCDE,
    "neural_sde": NeuralSDE,

    "hamiltonian_nn": HamiltonianNeuralNetwork,
    "hnn": HamiltonianNeuralNetwork,

    "symplectic_ode_net": SymplecticODENet,
    "symplectic_rnn": SymplecticRNN,

    "bayesian_rnn": BayesianRNN,
    "deep_state_space_model": DeepStateSpaceModel,
    "dssm": DeepStateSpaceModel,

    "neural_gp": NeuralGaussianProcess,
    "ngp": NeuralGaussianProcess,
}

_DYNAMICS_MODELS = (NeuralODE, LatentODE, ODERNN, NeuralCDE, NeuralSDE, SymplecticRNN)  # forward(state, t, ...)
_SEQUENCE_MODELS = (BayesianRNN, DeepStateSpaceModel)  # forward(x: (B,T,D), ...)
# HamiltonianNeuralNetwork/SymplecticODENet(z: (B,2*dim_q)) and
# NeuralGaussianProcess(x: (...,in_dim)) are genuinely per-point/pointwise
# (hamiltonian.py:28, symplectic_ode.py:68, neural_gp.py:202) — fall through.


def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        if cls in _DYNAMICS_MODELS:
            return {"input_kind": "dynamics", "expects": ["x0", "t"], "predicts": ["u"]}
        if cls in _SEQUENCE_MODELS:
            return {"input_kind": "sequence", "expects": ["x_past"], "predicts": ["u"]}
        return {"input_kind": "pointwise_coords", "expects": ["x"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="continuous", capabilities_getter=capabilities)

@dataclass
class ContinuousCatalog:
    registry: Dict[str, Type[ContinuousModelBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[ContinuousModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown continuous model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> ContinuousModelBase:
        return self.get(name)(**kwargs)
