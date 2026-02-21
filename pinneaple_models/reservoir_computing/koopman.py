from __future__ import annotations
"""Koopman operator for linearized dynamics in lifted space."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


class KoopmanOperator(RCBase):
    """
    Koopman Operator (EDMD-style MVP):

    Learn a linear operator K in a lifted space:
      z = phi(x)
      z_{t+1} = K z_t
      x_{t} ~ decode(z_t)  (optional)

    MVP:
      - phi is random Fourier features (RBF-like) or polynomial lift
      - fit K by least squares: K = Z_{t+1} * pinv(Z_t)
      - prediction by iterating K

    Inputs:
      x: (B,T,in_dim)
    Outputs:
      y: predicted x (B,T,out_dim==in_dim by default)
    """
    def __init__(
        self,
        in_dim: int,
        *,
        lift_dim: int = 1024,
        lift: Literal["rff", "poly"] = "rff",
        poly_degree: int = 2,
        rff_lengthscale: float = 1.0,
        l2: float = 1e-6,
        use_bias: bool = True,
        freeze_lift: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.lift_dim = int(lift_dim)
        self.lift = (lift or "rff").lower().strip()
        self.poly_degree = int(poly_degree)
        self.rff_lengthscale = float(rff_lengthscale)
        self.l2 = float(l2)
        self.use_bias = bool(use_bias)

        if self.lift not in ("rff", "poly"):
            raise ValueError("lift must be 'rff' or 'poly'")

        if self.lift == "rff":
            W = torch.randn(self.lift_dim, self.in_dim) * (1.0 / max(self.rff_lengthscale, 1e-12))
            b = torch.rand(self.lift_dim) * (2.0 * torch.pi)
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
            if freeze_lift:
                self.W.requires_grad_(False)
                self.b.requires_grad_(False)

            lift_out = self.lift_dim + (1 if self.use_bias else 0)

        else:
            # poly: [x, x^2, ..., x^k] (+bias)
            lift_out = self.in_dim * self.poly_degree + (1 if self.use_bias else 0)
            self.W = None
            self.b = None

        # Koopman matrix K: (F,F)
        self.K = nn.Parameter(torch.zeros(lift_out, lift_out), requires_grad=False)

        # simple linear decoder from lifted z to x
        self.decoder = nn.Linear(lift_out, self.in_dim, bias=False)
        self.decoder.weight.requires_grad_(False)

        self._fitted = False

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,in_dim) -> (N,F)
        feats = []
        if self.lift == "rff":
            # sqrt(2/m) cos(Wx + b)
            z = (2.0 / self.lift_dim) ** 0.5 * torch.cos(x @ self.W.t() + self.b)
            feats.append(z)
        else:
            p = x
            feats.append(p)
            for _ in range(2, self.poly_degree + 1):
                p = p * x
                feats.append(p)
        if self.use_bias:
            feats.append(torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "KoopmanOperator":
        """
        x: (B,T,in_dim) trajectory batch.
        Fit K using EDMD: Z1 ~ K Z0
        """
        B, T, D = x.shape
        x0 = x[:, :-1, :].reshape(-1, D)
        x1 = x[:, 1:, :].reshape(-1, D)

        Z0 = self._lift(x0)  # (N,F)
        Z1 = self._lift(x1)  # (N,F)

        # ridge regression for K: minimize ||Z1 - Z0 K^T||^2
        # Solve K^T = (Z0^T Z0 + l2 I)^-1 Z0^T Z1
        KT = self.ridge_solve(Z0, Z1, l2=self.l2)  # (F,F) but this is (F -> F) mapping
        K = KT.t()
        self.K.copy_(K)

        # decoder: best linear map from Z to x
        Wdec = self.ridge_solve(Z0, x0, l2=self.l2)  # (F, D)
        self.decoder.weight.copy_(Wdec.t())

        self._fitted = True
        return self

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x0: (B,in_dim)
        returns x_hat: (B,steps+1,in_dim)
        """
        B, D = x0.shape
        z = self._lift(x0)  # (B,F)

        xs = [x0]
        for _ in range(int(steps)):
            z = z @ self.K.t()
            xhat = self.decoder(z)
            xs.append(xhat)

        return torch.stack(xs, dim=1)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,in_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RCOutput:
        B, T, D = x.shape
        y_hat = self.rollout(x[:, 0, :], steps=T - 1)  # (B,T,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})
