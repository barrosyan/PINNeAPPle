# uno.py
from __future__ import annotations
"""U-Net neural operator for multi-scale operator learning."""
import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class UniversalNeuralOperator(NeuralOperatorBase):
    """
    Universal Operator Network (MVP):
      - learned latent representation
      - coordinate-aware decoding
    """
    def __init__(self, latent_dim: int, coord_dim: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + coord_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, z, coords, *, y_true=None, return_loss=False):
        h = self.encoder(z)
        h = h[:, None, :].expand(-1, coords.shape[0], -1)
        inp = torch.cat([h, coords[None, :, :].expand(z.shape[0], -1, -1)], dim=-1)
        y = self.decoder(inp)

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={})
