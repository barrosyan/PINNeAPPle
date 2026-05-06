from __future__ import annotations
"""SIREN: Sinusoidal Representation Networks for physics-informed learning.

Reference: Sitzmann et al., NeurIPS 2020
  "Implicit Neural Representations with Periodic Activation Functions"
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel, ModelOutput


class SineLayer(nn.Module):
    """Single sinusoidal layer with principled weight initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 1 / self.in_features
                )
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(BaseModel):
    """Sinusoidal Representation Network for physics-informed learning.

    Ideal for wave equations, Helmholtz, and high-frequency PDE fields.
    Every hidden layer uses ``sin(omega_0 * W x + b)`` as activation.
    The principled weight initialisation (Sitzmann et al. 2020) ensures that,
    at initialisation, each layer output is approximately unit-normally
    distributed, enabling stable gradient flow for deep networks.

    Reference:
        Sitzmann et al., NeurIPS 2020 —
        "Implicit Neural Representations with Periodic Activation Functions"

    Args:
        in_dim: Number of input coordinates.
        out_dim: Number of output fields.
        hidden_dim: Width of hidden layers.
        n_layers: Total number of layers (including input + output).
        omega_0: Frequency multiplier for sinusoidal activations.
        outermost_linear: If True, the last layer is a plain linear layer
            (no sine activation). Recommended for regression tasks.
        final_activation: Optional activation to apply after the last layer.
    """

    family: str = "pinns"
    name: str = "siren"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 5,
        omega_0: float = 30.0,
        outermost_linear: bool = True,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2 (at least one hidden layer + output)")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.omega_0 = omega_0
        self.outermost_linear = outermost_linear
        self.final_activation = final_activation

        layers: list[nn.Module] = []

        # First layer
        layers.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))

        # Output layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_dim, out_dim)
            with torch.no_grad():
                bound = math.sqrt(6 / hidden_dim) / omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0))

        self.net = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.

        Returns:
            :class:`~pinneaple_models.base.ModelOutput` with ``y`` of shape
            ``(..., out_dim)``.
        """
        h = x
        for layer in self.net:
            h = layer(h)
        if self.final_activation is not None:
            h = self.final_activation(h)
        return ModelOutput(y=h)
