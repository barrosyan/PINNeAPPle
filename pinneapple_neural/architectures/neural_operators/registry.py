from __future__ import annotations
"""Registry and catalog for neural operator model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import NeuralOperatorBase
from .deeponet import DeepONet
from .fno import FourierNeuralOperator
from .gno import GalerkinNeuralOperator
from .ms_deeponet import MultiScaleDeepONet
from .pino import PhysicsInformedNeuralOperator
from .uno import UniversalUNO


_REGISTRY: Dict[str, Type[NeuralOperatorBase]] = {
    "deeponet": DeepONet,
    "multiscale_deeponet": MultiScaleDeepONet,

    "fno": FourierNeuralOperator,
    "fourier_neural_operator": FourierNeuralOperator,

    "gno": GalerkinNeuralOperator,
    "galerkin_neural_operator": GalerkinNeuralOperator,

    "pino": PhysicsInformedNeuralOperator,
    "physics_informed_neural_operator": PhysicsInformedNeuralOperator,

    "uno": UniversalUNO,
    "universal_operator_network": UniversalUNO,
}

# Noether (Emmi AI) models — registered lazily so emmiai-noether is optional
try:
    from .noether_bridge import NOETHER_REGISTRY
    _REGISTRY.update(NOETHER_REGISTRY)
except Exception:
    pass

_NOETHER_KEYS = {
    "upt", "abupt", "anchored_branched_upt", "transformer", "transolver",
    "aero_upt", "aero_abupt", "aero_transformer", "aero_transolver",
}


def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        key = name.lower().strip()
        caps = {"predicts": ["u"], "supports_physics_loss": (key in ("pino", "physics_informed_neural_operator"))}

        if key in ("deeponet", "multiscale_deeponet"):
            caps.update({"input_kind": "operator_branch_trunk", "expects": ["u_branch", "coords"]})
            return caps

        if key in ("fno", "fourier_neural_operator"):
            caps.update({"input_kind": "grid_1d", "expects": ["u_grid_1d"]})
            return caps

        if key in ("gno", "galerkin_neural_operator"):
            caps.update({"input_kind": "operator_branch_trunk", "expects": ["u_points", "coords_points"]})
            return caps

        if key in ("uno", "universal_operator_network"):
            caps.update({"input_kind": "grid", "expects": ["u_grid"]})
            return caps

        if key in ("pino", "physics_informed_neural_operator"):
            caps.update({"input_kind": "operator_branch_trunk", "expects": ["u", "physics_fn", "physics_data"]})
            return caps

        if key.startswith("noether") or key in _NOETHER_KEYS:
            # Mesh/point-cloud transformers (UPT/AB-UPT/Transolver) — closer to
            # graph/mesh models than plain coords; also need the optional
            # emmiai-noether dependency (see noether_bridge.py's _import_noether()).
            caps.update({"input_kind": "graph", "expects": ["graph"]})
            return caps

        caps.update({"input_kind": "grid", "expects": ["u"]})
        return caps

    register_family_registry(_REGISTRY, family="neural_operators", capabilities_getter=capabilities)

@dataclass
class NeuralOperatorCatalog:
    registry: Dict[str, Type[NeuralOperatorBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[NeuralOperatorBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown neural operator '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> NeuralOperatorBase:
        return self.get(name)(**kwargs)
