from __future__ import annotations
"""Registry and catalog for autoencoder model family."""

from dataclasses import dataclass
from typing import Any, Dict, Type

from .base import AEBase
from .dense_ae import DenseAutoencoder
from .ae_2d import Autoencoder2D
from .vae import VariationalAutoencoder
from .kae import KAEAutoencoder
from .koopman_pi_ae import PhysicsInformedKoopmanAutoencoder
from .ae_rom_hybrid import AEROMHybrid


_REGISTRY: Dict[str, Type[AEBase]] = {
    "dense_ae": DenseAutoencoder,
    "dense": DenseAutoencoder,
    "ae": DenseAutoencoder,

    "autoencoder_2d": Autoencoder2D,
    "ae_2d": Autoencoder2D,

    "vae": VariationalAutoencoder,
    "variational_autoencoder": VariationalAutoencoder,

    "kae": KAEAutoencoder,
    "kernel_ae": KAEAutoencoder,

    "pi_koopman_ae": PhysicsInformedKoopmanAutoencoder,
    "koopman_pi_ae": PhysicsInformedKoopmanAutoencoder,

    "ae_rom_hybrid": AEROMHybrid,
    "rom_hybrid": AEROMHybrid,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="autoencoders")

@dataclass
class AutoencoderCatalog:
    """
    Group entrypoint:
      cat = AutoencoderCatalog()
      model = cat.build("vae", input_dim=..., latent_dim=...)
    """
    registry: Dict[str, Type[AEBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[AEBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown autoencoder '{name}'. Available: {self.list()[:20]} ...")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> AEBase:
        cls = self.get(name)
        return cls(**kwargs)
