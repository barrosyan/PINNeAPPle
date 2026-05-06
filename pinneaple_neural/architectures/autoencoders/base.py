from __future__ import annotations
"""Base classes for autoencoder models."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn


@dataclass
class AEOutput:
    """
    Standardized output for all AEs.
    """
    x_hat: torch.Tensor
    z: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class AEBase(nn.Module):
    """
    Base class for all autoencoders in Pinneaple.

    Contract:
      - encode(x) -> z
      - decode(z) -> x_hat
      - forward(x) -> AEOutput
      - loss(output, x) -> dict of losses (recon + model-specific)
    """
    def __init__(self):
        super().__init__()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> AEOutput:
        z = self.encode(x)
        x_hat = self.decode(z)
        losses = self.loss_from_parts(x_hat=x_hat, z=z, x=x)
        return AEOutput(x_hat=x_hat, z=z, losses=losses, extras={})

    def loss_from_parts(self, *, x_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # default: reconstruction only
        recon = torch.mean((x_hat - x) ** 2)
        return {"recon": recon, "total": recon}
