from __future__ import annotations
"""Radial basis function network for reservoir-style regression."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


class RBFNetwork(RCBase):
    """
    RBF Network:
      - Centers C (learned or fixed)
      - Features: phi_i(x) = exp(-||x-C_i||^2 / (2*sigma^2))
      - Output weights solved by ridge regression (MVP)

    Fit:
      - If centers not provided: sample from x (simple init)
      - sigma based on median distance heuristic (optional)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_centers: int = 512,
        sigma: Optional[float] = None,
        l2: float = 1e-6,
        learn_centers: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_centers = int(num_centers)
        self.l2 = float(l2)

        self.centers = nn.Parameter(torch.randn(self.num_centers, self.in_dim))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
        if sigma is not None:
            self.log_sigma.data = torch.log(torch.tensor(float(sigma)).clamp_min(1e-12))

        if not learn_centers:
            self.centers.requires_grad_(False)

        self.W_out = nn.Parameter(torch.zeros(self.num_centers, self.out_dim), requires_grad=False)
        self._fitted = False

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,in_dim), centers:(M,in_dim) -> (N,M)
        x2 = (x ** 2).sum(dim=-1, keepdim=True)             # (N,1)
        c2 = (self.centers ** 2).sum(dim=-1)[None, :]      # (1,M)
        xc = x @ self.centers.t()                           # (N,M)
        dist2 = x2 + c2 - 2.0 * xc
        sigma2 = torch.exp(2.0 * self.log_sigma).clamp_min(1e-12)
        return torch.exp(-0.5 * dist2 / sigma2)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, *, init_centers_from_data: bool = True) -> "RBFNetwork":
        """
        x: (N,in_dim), y: (N,out_dim)
        """
        N = x.shape[0]
        if init_centers_from_data:
            idx = torch.randperm(N, device=x.device)[: self.num_centers]
            self.centers.copy_(x[idx].detach().clone())

        # median heuristic for sigma if user didn't set it (log_sigma close to 0 means sigma=1)
        # Here we only adjust if sigma is effectively default (close to 1) and data suggests different scale.
        # This keeps things stable.
        if self.log_sigma.detach().abs().item() < 1e-6 and N >= 2:
            # sample distances
            ii = torch.randperm(N, device=x.device)[: min(N, 256)]
            jj = torch.randperm(N, device=x.device)[: min(N, 256)]
            d = (x[ii] - x[jj]).pow(2).sum(dim=-1).sqrt()
            med = torch.median(d).clamp_min(1e-6)
            self.log_sigma.copy_(torch.log(med))

        Phi = self._phi(x)  # (N,M)
        W = self.ridge_solve(Phi, y, l2=self.l2)
        self.W_out.copy_(W)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> RCOutput:
        Phi = self._phi(x)
        y = Phi @ self.W_out

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y, losses=losses, extras={"sigma": torch.exp(self.log_sigma).item(), "fitted": self._fitted})
