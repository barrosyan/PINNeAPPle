from __future__ import annotations
"""Modified MLP with Fourier feature embedding and highway (U/V gating) connections.

Reference: Wang et al. 2022
  "Improved Architectures and Training Algorithms for Deep Operator Networks"
  (also called "Modified MLP" or "Fourier-feature highway MLP" in the PINN literature)
"""

import math
from typing import Type

import torch
import torch.nn as nn

from .base import BaseModel, ModelOutput


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier feature embedding for coordinate inputs.

    Maps ``x`` to ``[cos(2π B x), sin(2π B x)]`` where ``B`` is a random
    (or learnable) projection matrix, expanding a low-dimensional coordinate
    into a high-dimensional feature that exposes oscillatory structure to the
    network.

    Args:
        in_dim: Input coordinate dimension.
        n_fourier: Number of random Fourier features (output dim = 2 * n_fourier).
        sigma: Standard deviation for the random projection matrix.
        trainable: If True, ``B`` is a learnable parameter; otherwise it is a
            fixed buffer.
    """

    def __init__(
        self,
        in_dim: int,
        n_fourier: int = 64,
        sigma: float = 1.0,
        trainable: bool = False,
    ):
        super().__init__()
        B = torch.randn(in_dim, n_fourier) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.out_dim = 2 * n_fourier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B  # (..., n_fourier)
        return torch.cat(
            [torch.cos(2 * math.pi * proj), torch.sin(2 * math.pi * proj)],
            dim=-1,
        )


class ModifiedMLP(BaseModel):
    """Modified MLP with Fourier feature embedding and highway U/V connections.

    Architecture:
    1. Input ``x`` is mapped to a Fourier feature embedding ``h``.
    2. Two encoder branches ``U = σ(W_u h + b_u)`` and
       ``V = σ(W_v h + b_v)`` produce context vectors.
    3. Each hidden layer applies element-wise highway gating::

           z_new = σ(W z + b)
           z_new = z_new * U + (1 - z_new) * V

    4. A final linear projection maps to ``out_dim``.

    This design (Wang et al. 2022) substantially improves convergence for
    physics-informed operator networks and PINNs by maintaining information
    flow from the input encoding throughout all layers.

    Args:
        in_dim: Number of input coordinates.
        out_dim: Number of output fields.
        hidden_dim: Width of hidden layers and encoder branches.
        n_layers: Total number of hidden layers (depth of the gated stack).
        n_fourier: Number of random Fourier features.
        sigma: Bandwidth for the Fourier feature projection.
        activation: Activation class (called with no arguments) used in
            encoder branches and hidden layers.  Defaults to ``nn.Tanh``.
        trainable_fourier: If True, the Fourier projection matrix is learnable.
    """

    family: str = "pinns"
    name: str = "modified_mlp"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 6,
        n_fourier: int = 32,
        sigma: float = 1.0,
        activation: Type[nn.Module] = nn.Tanh,
        trainable_fourier: bool = False,
    ):
        super().__init__()
        self.embed = FourierFeatureEmbedding(in_dim, n_fourier, sigma, trainable=trainable_fourier)
        embed_dim = self.embed.out_dim

        # Two encoder branches that provide global context
        self.U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), activation())
        self.V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), activation())

        # Gated hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(max(n_layers - 1, 1))]
        )
        self.act = activation()
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.

        Returns:
            :class:`~pinneaple_models.base.ModelOutput` with ``y`` of shape
            ``(..., out_dim)``.
        """
        h = self.embed(x)
        u = self.U(h)   # global context branch U
        v = self.V(h)   # global context branch V

        z = u  # initialise hidden state from U branch
        for linear in self.layers:
            z = self.act(linear(z))
            z = z * u + (1 - z) * v  # element-wise highway gating

        return ModelOutput(y=self.output(z))
