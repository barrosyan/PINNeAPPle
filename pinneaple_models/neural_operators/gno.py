from __future__ import annotations
"""Graph neural operator for mesh-based PDEs."""
import torch
import torch.nn as nn

from .base import NeuralOperatorBase, OperatorOutput


class GalerkinNeuralOperator(NeuralOperatorBase):
    """
    GNO MVP:
      - Projects input onto learned basis
      - Applies operator in coefficient space
      - Reconstructs function
    """
    def __init__(self, in_dim: int, out_dim: int, basis_dim: int = 64):
        super().__init__()
        self.encoder = nn.Linear(in_dim, basis_dim)
        self.operator = nn.Linear(basis_dim, basis_dim)
        self.decoder = nn.Linear(basis_dim, out_dim)

    def forward(self, u, *, y_true=None, return_loss=False):
        coeffs = self.encoder(u)
        coeffs = torch.nn.functional.gelu(self.operator(coeffs))
        y = self.decoder(coeffs)

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={"coeffs": coeffs})
