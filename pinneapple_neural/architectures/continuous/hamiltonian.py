from __future__ import annotations
"""Hamiltonian neural network for energy-conserving dynamics."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class HamiltonianNeuralNetwork(ContinuousModelBase):
    """
    HNN MVP:
      - Learn Hamiltonian H(q,p)
      - Dynamics: dq/dt = dH/dp, dp/dt = -dH/dq
    """
    def __init__(self, dim_q: int, hidden: int = 128, num_layers: int = 3):
        super().__init__()
        self.dim_q = int(dim_q)
        self.dim_p = int(dim_q)

        layers = [nn.Linear(2 * self.dim_q, hidden), nn.Tanh()]
        for _ in range(max(0, int(num_layers) - 1)):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.H = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,  # (B, 2*dim_q) = [q, p]
        *,
        y_true: Optional[torch.Tensor] = None,  # (B, 2*dim_q)
        return_loss: bool = False,
    ) -> ContOutput:
        # Make z a leaf tensor requiring grad (safe for autograd.grad)
        z = z.detach().clone().requires_grad_(True)

        # Per-sample Hamiltonian
        H_batch = self.H(z)  # (B, 1)

        # dH/dz for each sample
        grad = torch.autograd.grad(
            outputs=H_batch,
            inputs=z,
            grad_outputs=torch.ones_like(H_batch),
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, 2*dim_q)

        qdot = grad[:, self.dim_q : self.dim_q + self.dim_p]
        pdot = -grad[:, : self.dim_q]
        dz = torch.cat([qdot, pdot], dim=-1)  # (B, 2*dim_q)

        losses: Dict[str, torch.Tensor] = {
            "total": torch.zeros((), device=z.device, dtype=dz.dtype)
        }
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(dz, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(
            y=dz,
            losses=losses,
            extras={
                "H": H_batch,
                "H_sum": H_batch.sum(),
            },
        )
