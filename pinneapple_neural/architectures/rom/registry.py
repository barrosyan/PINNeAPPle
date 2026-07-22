from __future__ import annotations
"""Registry and catalog for the reduced-order model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import ROMBase
from .pod import POD
from .dmd import DynamicModeDecomposition
from .havok import HAVOK
from .opinf import OperatorInference
from .rom_hybrid import ROMHybrid
from .deep_uq_rom import DeepUQROM
from .sindy import SINDy
from .koopman import KoopmanAutoencoder
from .neural_rom import NeuralROM


_REGISTRY: Dict[str, Type[ROMBase]] = {
    # Projection-based
    "pod":                         POD,
    "proper_orthogonal_decomposition": POD,

    # Linear dynamics
    "dmd":                         DynamicModeDecomposition,
    "dynamic_mode_decomposition":  DynamicModeDecomposition,
    "havok":                       HAVOK,

    # Data-driven operators
    "operator_inference":          OperatorInference,
    "opinf":                       OperatorInference,

    # Sparse identification
    "sindy":                       SINDy,
    "sparse_identification":       SINDy,

    # Deep learning ROMs
    "koopman_rom":                 KoopmanAutoencoder,
    "koopman_autoencoder":         KoopmanAutoencoder,
    "deep_koopman":                KoopmanAutoencoder,

    "neural_rom":                  NeuralROM,
    "nrom":                        NeuralROM,

    # Hybrid / UQ
    "rom_hybrid":                  ROMHybrid,
    "deep_uq_rom":                 DeepUQROM,
    "uq_rom":                      DeepUQROM,
}


def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        # Reduced-order models compress a *distribution* of field snapshots
        # (POD/DMD/SINDy/Koopman all take X: (N, state_dim) full-field state
        # vectors, see rom/*.py forward()s) — not coords->solution regression.
        return {"input_kind": "autoencoder", "expects": ["x"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="rom", capabilities_getter=capabilities)


@dataclass
class ROMCatalog:
    registry: Dict[str, Type[ROMBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[ROMBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown ROM model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> ROMBase:
        return self.get(name)(**kwargs)
