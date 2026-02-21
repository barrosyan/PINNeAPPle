from __future__ import annotations
"""Registry and catalog for transformer time series model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import TimeSeriesModelBase
from .transformer import VanillaTransformer
from .informer import Informer
from .tft import TemporalFusionTransformer
from .autoformer import Autoformer
from .fedformer import FEDformer
from .timesnet import TimesNet


_REGISTRY: Dict[str, Type[TimeSeriesModelBase]] = {
    "transformer": VanillaTransformer,
    "vanilla_transformer": VanillaTransformer,

    "informer": Informer,

    "tft": TemporalFusionTransformer,
    "temporal_fusion_transformer": TemporalFusionTransformer,

    "autoformer": Autoformer,
    "fedformer": FEDformer,

    "timesnet": TimesNet,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="transformers")

@dataclass
class TransformerCatalog:
    registry: Dict[str, Type[TimeSeriesModelBase]] | None = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[TimeSeriesModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown transformer model '{name}'. Available: {self.list()[:20]} ...")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> TimeSeriesModelBase:
        cls = self.get(name)
        return cls(**kwargs)
