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

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="continuous")

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
