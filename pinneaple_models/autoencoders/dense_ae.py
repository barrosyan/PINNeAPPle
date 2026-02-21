from __future__ import annotations
"""Dense autoencoder with MLP encoder-decoder."""

from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn

from .base import AEBase


def _mlp(dims: List[int], act: nn.Module, last_act: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act)
    return nn.Sequential(*layers)


class DenseAutoencoder(AEBase):
    """
    Plain MLP autoencoder for vector inputs.

    Args:
      input_dim: flattened input dimension
      latent_dim: bottleneck dim
      hidden: list of hidden widths for encoder (decoder mirrors)
      activation: "tanh"|"relu"|"gelu"
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(activation.lower(), nn.GELU())

        enc_dims = [input_dim, *list(hidden), latent_dim]
        dec_dims = [latent_dim, *list(reversed(hidden)), input_dim]

        self.encoder = _mlp(enc_dims, act, last_act=False)
        self.decoder = _mlp(dec_dims, act, last_act=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
