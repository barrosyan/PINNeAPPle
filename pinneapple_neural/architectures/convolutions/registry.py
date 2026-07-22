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

_GRID_RANK = {
    Conv1DModel: "grid_1d",
    Conv2DModel: "grid_2d",
    Conv3DModel: "grid_3d",
}


def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        # All expect a channel-first rasterized grid tensor, not scattered
        # (N, in_dim) points.
        return {"input_kind": _GRID_RANK.get(cls, "grid"), "expects": ["u_grid"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="convolutions", capabilities_getter=capabilities)

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
