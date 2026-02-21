from __future__ import annotations
"""Registry and catalog for classical time series model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import ClassicalTSBase
from .var import VAR
from .arima import ARIMA
from .kalman import KalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .enkf import EnsembleKalmanFilter
from .tcn import TCN


_REGISTRY: Dict[str, Type[ClassicalTSBase]] = {
    "var": VAR,
    "vector_autoregression": VAR,

    "arima": ARIMA,

    "kalman_filter": KalmanFilter,
    "kf": KalmanFilter,

    "extended_kalman_filter": ExtendedKalmanFilter,
    "ekf": ExtendedKalmanFilter,

    "unscented_kalman_filter": UnscentedKalmanFilter,
    "ukf": UnscentedKalmanFilter,

    "ensemble_kalman_filter": EnsembleKalmanFilter,
    "enkf": EnsembleKalmanFilter,

    "tcn": TCN,
    "temporal_convolutional_network": TCN,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="classical_ts")

@dataclass
class ClassicalTSCatalog:
    registry: Dict[str, Type[ClassicalTSBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[ClassicalTSBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown classical_ts model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> ClassicalTSBase:
        return self.get(name)(**kwargs)
