from __future__ import annotations
"""Vector autoregression model for multivariate time series."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ClassicalTSBase, ClassicalTSOutput


class VAR(ClassicalTSBase):
    """
    VAR(p) with ridge regression (closed form).

    x_t = c + sum_{k=1..p} A_k x_{t-k} + eps

    Fit uses batch-aggregated least squares over all sequences.
    """
    def __init__(self, dim: int, p: int = 1, l2: float = 1e-6, use_bias: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.l2 = float(l2)
        self.use_bias = bool(use_bias)

        feat_dim = self.dim * self.p + (1 if self.use_bias else 0)
        self.W = nn.Parameter(torch.zeros(feat_dim, self.dim), requires_grad=False)
        self._fitted = False

    def _make_design(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized design matrix builder.

        x: (B, T, D)
        returns X: (B*(T-p), D*p (+1))
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B,T,D). Got shape={tuple(x.shape)}")
        B, T, D = x.shape
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got D={D}")
        if T <= self.p:
            raise ValueError(f"T must be > p. Got T={T}, p={self.p}")

        # IMPORTANT: for x (B,T,D), unfold along T produces (B, T-p, D, p+1)
        win = x.unfold(dimension=1, size=self.p + 1, step=1)  # (B, T-p, D, p+1)

        # lags are the first p elements of the window, target is the last one
        lags = win[..., :-1]               # (B, T-p, D, p)
        lags = lags.flip(dims=[-1])        # newest first along window axis -> (B, T-p, D, p)

        # reshape expects (B, T-p, p, D) so that features are [x_{t-1}||...||x_{t-p}]
        lags = lags.permute(0, 1, 3, 2).contiguous()  # (B, T-p, p, D)

        X = lags.reshape(B * (T - self.p), self.p * D)  # (B*(T-p), D*p)

        if self.use_bias:
            ones = torch.ones((X.shape[0], 1), device=x.device, dtype=x.dtype)
            X = torch.cat([X, ones], dim=-1)

        return X

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized targets builder.

        x: (B, T, D)
        returns Y: (B*(T-p), D)
        """
        B, T, D = x.shape
        if T <= self.p:
            raise ValueError(f"T must be > p. Got T={T}, p={self.p}")

        win = x.unfold(dimension=1, size=self.p + 1, step=1)  # (B, T-p, D, p+1)
        Y = win[..., -1]  # (B, T-p, D)
        Y = Y.reshape(B * (T - self.p), D)
        return Y

    @staticmethod
    def _ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float) -> torch.Tensor:
        """
        Ridge regression solve using a more numerically stable method than normal equations solve.
        Uses Cholesky on (X^T X + l2 I), which is SPD for l2 > 0.
        """
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        A = X.T @ X + float(l2) * I
        B = X.T @ Y

        # Cholesky factorization + solve (more stable than torch.linalg.solve on normal equations)
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(B, L)
        return W

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "VAR":
        X = self._make_design(x)
        Y = self._targets(x)
        W = self._ridge_solve(X, Y, self.l2)
        self.W.copy_(W)
        self._fitted = True
        return self

    @torch.no_grad()
    def forecast(self, x_hist: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x_hist: (B, T, D) with T >= p
        returns: (B, steps, D)
        """
        if not self._fitted:
            raise RuntimeError("VAR not fitted. Call fit(x) first.")

        B, T, D = x_hist.shape
        if T < self.p:
            raise ValueError(f"Need at least p={self.p} history steps, got {T}.")
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got D={D}")

        buf = [x_hist[:, -k, :] for k in range(1, self.p + 1)]  # newest first, each (B,D)
        preds = []

        for _ in range(int(steps)):
            feat = torch.cat(buf, dim=-1)  # (B, D*p)
            if self.use_bias:
                feat = torch.cat(
                    [feat, torch.ones((B, 1), device=x_hist.device, dtype=x_hist.dtype)],
                    dim=-1
                )
            y = feat @ self.W  # (B,D)
            preds.append(y)
            buf = [y] + buf[:-1]

        return torch.stack(preds, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> ClassicalTSOutput:
        if not self._fitted:
            raise RuntimeError("VAR not fitted. Call fit(x) first.")

        # one-step ahead prediction over the provided sequence
        B, T, D = x.shape
        if T <= self.p:
            raise ValueError(f"T must be > p. Got T={T}, p={self.p}")
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got D={D}")

        X = self._make_design(x)  # (B*(T-p), F)
        Yhat = X @ self.W         # (B*(T-p), D)

        y_hat = Yhat.reshape(T - self.p, B, D).permute(1, 0, 2)  # (B, T-p, D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if y_true.shape != x.shape:
                raise ValueError(f"y_true must have same shape as x. Got {tuple(y_true.shape)} vs {tuple(x.shape)}")
            losses["mse"] = torch.mean((y_hat - y_true[:, self.p:, :]) ** 2)
            losses["total"] = losses["mse"]

        return ClassicalTSOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})
