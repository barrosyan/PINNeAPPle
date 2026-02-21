from __future__ import annotations
"""Hybrid RBF network combining linear and nonlinear features."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import RCBase, RCOutput
from .rbf import RBFNetwork


class HybridRBFNetwork(RCBase):
    """
    Hybrid RBF Network:
      y = Phi(x) W_rbf + x W_lin + b
    where W_* solved by ridge regression in a single linear system.

    This is often much stronger than pure RBF on real physics data.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_centers: int = 512,
        sigma: Optional[float] = None,
        l2: float = 1e-6,
        learn_centers: bool = False,
        use_linear: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.l2 = float(l2)
        self.use_linear = bool(use_linear)
        self.use_bias = bool(use_bias)

        self.rbf = RBFNetwork(
            in_dim=in_dim,
            out_dim=out_dim,
            num_centers=num_centers,
            sigma=sigma,
            l2=l2,
            learn_centers=learn_centers,
        )

        # combined weights (features_dim, out_dim)
        feat_dim = num_centers + (in_dim if self.use_linear else 0) + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(feat_dim, out_dim), requires_grad=False)
        self._fitted = False

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        Phi = self.rbf._phi(x)  # (N,M)
        feats = [Phi]
        if self.use_linear:
            feats.append(x)
        if self.use_bias:
            feats.append(torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, *, init_centers_from_data: bool = True) -> "HybridRBFNetwork":
        # init centers + sigma via underlying RBF
        self.rbf.fit(x, y, init_centers_from_data=init_centers_from_data)

        F = self._features(x)  # (N,feat_dim)
        W = self.ridge_solve(F, y, l2=self.l2)
        self.W_out.copy_(W)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False) -> RCOutput:
        F = self._features(x)
        y = F @ self.W_out

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y, losses=losses, extras={"fitted": self._fitted})
