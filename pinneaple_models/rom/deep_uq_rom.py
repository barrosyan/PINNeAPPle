from __future__ import annotations
"""Deep UQ ROM with uncertainty quantification."""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ROMBase, ROMOutput


class DeepUQROM(ROMBase):
    """
    Deep UQ-ROM (MVP scaffold):
      - A small latent predictor that outputs mean and log-variance.

    Intended use:
      - plug after POD encoder
      - train with NLL (Gaussian)

    Inputs:
      a: (B,T,r) latent
    Outputs:
      a_hat_mean, a_hat_logvar
    """
    def __init__(self, r: int, hidden: int = 256, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.r = int(r)

        net = []
        in_dim = r
        for _ in range(int(layers)):
            net += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(float(dropout))]
            in_dim = hidden
        self.net = nn.Sequential(*net)
        self.mean = nn.Linear(hidden, r)
        self.logvar = nn.Linear(hidden, r)

    def forward(self, a: torch.Tensor, *, a_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> ROMOutput:
        # a: (B,T,r) -> predict next-step in a teacher-forcing style (shifted)
        B, T, r = a.shape
        inp = a[:, :-1, :].reshape(-1, r)
        h = self.net(inp)
        mu = self.mean(h).view(B, T - 1, r)
        lv = self.logvar(h).view(B, T - 1, r)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=a.device)}
        if return_loss and a_true is not None:
            tgt = a_true[:, 1:, :]
            # Gaussian NLL
            var = torch.exp(lv).clamp_min(1e-8)
            nll = 0.5 * (torch.log(var) + ((tgt - mu) ** 2) / var)
            losses["nll"] = nll.mean()
            losses["total"] = losses["nll"]

        return ROMOutput(
            y=mu,
            losses=losses,
            extras={"logvar": lv},
        )
