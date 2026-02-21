from __future__ import annotations
"""Operator inference for data-driven reduced-order modeling."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput


class OperatorInference(ROMBase):
    """
    Operator Inference (OpInf) MVP:

    Fit latent dynamics:
      a_{t+1} = A a_t + H (a_t ⊗ a_t) + b

    - You provide latent trajectories a (e.g. from POD encoder)
    - Fit A, H, b by ridge regression

    Shapes:
      a: (B,T,r)
      A: (r,r)
      H: (r, r*r)
      b: (r,)
    """
    def __init__(self, r: int, l2: float = 1e-6, use_quadratic: bool = True, use_bias: bool = True):
        super().__init__()
        self.r = int(r)
        self.l2 = float(l2)
        self.use_quadratic = bool(use_quadratic)
        self.use_bias = bool(use_bias)

        feat_dim = r + (r * r if self.use_quadratic else 0) + (1 if self.use_bias else 0)
        self.W = nn.Parameter(torch.zeros(feat_dim, r), requires_grad=False)
        self._fitted = False

    def _features(self, a: torch.Tensor) -> torch.Tensor:
        # a: (N,r)
        feats = [a]
        if self.use_quadratic:
            # kron a⊗a (N, r*r)
            feats.append(torch.einsum("ni,nj->nij", a, a).reshape(a.shape[0], -1))
        if self.use_bias:
            feats.append(torch.ones((a.shape[0], 1), device=a.device, dtype=a.dtype))
        return torch.cat(feats, dim=-1)

    @staticmethod
    def _ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float) -> torch.Tensor:
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        return torch.linalg.solve(X.t() @ X + l2 * I, X.t() @ Y)

    @torch.no_grad()
    def fit(self, a: torch.Tensor) -> "OperatorInference":
        # a: (B,T,r)
        B, T, r = a.shape
        a0 = a[:, :-1, :].reshape(-1, r)
        a1 = a[:, 1:, :].reshape(-1, r)
        X = self._features(a0)
        self.W.copy_(self._ridge_solve(X, a1, self.l2))
        self._fitted = True
        return self

    @torch.no_grad()
    def rollout(self, a0: torch.Tensor, steps: int) -> torch.Tensor:
        # a0: (B,r) -> (B,steps+1,r)
        B, r = a0.shape
        cur = a0
        out = [cur]
        for _ in range(int(steps)):
            X = self._features(cur)  # (B,F)
            cur = X @ self.W
            out.append(cur)
        return torch.stack(out, dim=1)

    def forward(self, a: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        # a: (B,T,r)
        B, T, r = a.shape
        yhat = self.rollout(a[:, 0, :], steps=T - 1)  # (B,T,r)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=a.device)}
        if return_loss:
            losses["mse"] = self.mse(yhat, a)
            losses["total"] = losses["mse"]

        return ROMOutput(y=yhat, losses=losses, extras={"fitted": self._fitted})
