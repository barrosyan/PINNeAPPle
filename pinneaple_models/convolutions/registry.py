from __future__ import annotations
"""Registry and catalog for convolution model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import ConvModelBase
from .conv1d import Conv1DModel
from .conv2d import Conv2DModel
from .conv3d import Conv3DModel


_REGISTRY: Dict[str, Type[ConvModelBase]] = {
    "conv1d": Conv1DModel,
    "conv_1d": Conv1DModel,

    "conv2d": Conv2DModel,
    "conv_2d": Conv2DModel,

    "conv3d": Conv3DModel,
    "conv_3d": Conv3DModel,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="convolutions")

@dataclass
class ConvolutionCatalog:
    registry: Dict[str, Type[ConvModelBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[ConvModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown convolution model '{name}'. Available: {self.list()[:20]} ...")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> ConvModelBase:
        cls = self.get(name)
        return cls(**kwargs)
