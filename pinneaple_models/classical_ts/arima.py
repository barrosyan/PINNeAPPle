from __future__ import annotations
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from .base import ClassicalTSBase, ClassicalTSOutput


class ARIMA(ClassicalTSBase):
    """
    ARIMA(p,d,q) MVP without external deps:

    - Supports differencing order d
    - Supports AR(p) with ridge regression
    - q (MA) is NOT implemented in this MVP (kept for API compatibility)

    Use:
      fit(x) where x: (B,T,dim) or (B,T,1)
      forecast(x_hist, steps)

    Notes:
      - For d>0, forward() returns predictions in ORIGINAL space (aligned), and also
        provides differenced predictions in extras["y_differenced"].
      - This is effectively an AR(p) on Δ^d(x), i.e., ARI(p,d) since MA(q) is not implemented.
    """

    def __init__(
        self,
        dim: int = 1,
        p: int = 3,
        d: int = 0,
        q: int = 0,
        l2: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)  # placeholder
        self.l2 = float(l2)
        self.use_bias = bool(use_bias)

        feat_dim = self.dim * self.p + (1 if self.use_bias else 0)
        self.W = nn.Parameter(torch.zeros(feat_dim, self.dim), requires_grad=False)
        self._fitted = False

    def _difference(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        y = x
        for _ in range(self.d):
            y = y[:, 1:, :] - y[:, :-1, :]
        return y

    def _last_diff_states(self, x_hist: torch.Tensor) -> List[torch.Tensor]:
        """
        Builds integration state for undifferencing up to order d-1.

        Returns:
          states = [x_T, Δx_T, Δ^2 x_T, ..., Δ^{d-1} x_T]  each (B,D)
        where T is the last time index in x_hist.

        For d=0: returns [].
        """
        if self.d == 0:
            return []

        if x_hist.shape[1] <= self.d:
            raise ValueError(
                f"Need at least d+1={self.d+1} points in x_hist to undifference, got T={x_hist.shape[1]}."
            )

        states: List[torch.Tensor] = []
        cur = x_hist  # (B,T,D)

        # order 0: last level
        states.append(cur[:, -1, :])

        # orders 1..d-1: last differences
        for _ in range(1, self.d):
            cur = cur[:, 1:, :] - cur[:, :-1, :]
            states.append(cur[:, -1, :])

        return states

    def _undifference_forecast(self, x_hist: torch.Tensor, dx_fore: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct forecast in original space given Δ^d-forecasts.

        Args:
          x_hist: (B,T,D) original history
          dx_fore: (B,steps,D) forecasts in differenced space of order d (i.e., Δ^d x)

        Returns:
          y: (B,steps,D) forecast in original space
        """
        if self.d == 0:
            return dx_fore

        # states: [x_T, Δx_T, ..., Δ^{d-1} x_T]
        states = self._last_diff_states(x_hist)

        y = []
        for t in range(dx_fore.shape[1]):
            top = dx_fore[:, t, :]  # Δ^d x at step t
            tmp = states + [top]    # length d+1

            # integrate down: Δ^{k-1} += Δ^k
            for k in range(self.d, 0, -1):
                tmp[k - 1] = tmp[k - 1] + tmp[k]

            states = tmp[:-1]  # keep [x, Δx, ..., Δ^{d-1}] for next step
            y.append(states[0])

        return torch.stack(y, dim=1)

    @staticmethod
    def _ridge_solve(X: torch.Tensor, Y: torch.Tensor, l2: float) -> torch.Tensor:
        F = X.shape[1]
        I = torch.eye(F, device=X.device, dtype=X.dtype)
        return torch.linalg.solve(X.t() @ X + l2 * I, X.t() @ Y)

    def _make_design(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        rows = []
        for t in range(self.p, T):
            lagged = [x[:, t - k, :] for k in range(1, self.p + 1)]
            row = torch.cat(lagged, dim=-1)
            if self.use_bias:
                row = torch.cat(
                    [row, torch.ones((B, 1), device=x.device, dtype=x.dtype)],
                    dim=-1,
                )
            rows.append(row)
        return torch.cat(rows, dim=0)

    def _targets(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, t, :] for t in range(self.p, x.shape[1])], dim=0)

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "ARIMA":
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[-1]}")

        xd = self._difference(x)
        if xd.shape[1] <= self.p:
            raise ValueError(f"Not enough timesteps after differencing: T'={xd.shape[1]} <= p={self.p}")

        X = self._make_design(xd)
        Y = self._targets(xd)
        self.W.copy_(self._ridge_solve(X, Y, self.l2))
        self._fitted = True
        return self

    @torch.no_grad()
    def forecast(self, x_hist: torch.Tensor, steps: int) -> torch.Tensor:
        if x_hist.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x_hist.shape[-1]}")

        xd = self._difference(x_hist)
        B, T, D = xd.shape
        if T < self.p:
            raise ValueError(f"Need at least p={self.p} points after differencing, got {T}.")

        buf = [xd[:, -k, :] for k in range(1, self.p + 1)]
        dx_preds = []
        for _ in range(int(steps)):
            feat = torch.cat(buf, dim=-1)
            if self.use_bias:
                feat = torch.cat(
                    [feat, torch.ones((B, 1), device=x_hist.device, dtype=x_hist.dtype)],
                    dim=-1,
                )
            dx = feat @ self.W
            dx_preds.append(dx)
            buf = [dx] + buf[:-1]

        dx_fore = torch.stack(dx_preds, dim=1)  # (B,steps,D)

        # undifference back to original (now correct for any d)
        return self._undifference_forecast(x_hist, dx_fore)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> ClassicalTSOutput:
        """
        One-step-ahead predictions along the provided sequence.

        Returns predictions in ORIGINAL space by default.
        Also exposes differenced predictions in extras["y_differenced"].
        """
        xd = self._difference(x)
        B, Tp, D = xd.shape

        if Tp <= self.p:
            raise ValueError(f"Not enough timesteps after differencing: T'={Tp} <= p={self.p}")

        X = self._make_design(xd)
        Yhat = X @ self.W
        yhat_d = Yhat.view(Tp - self.p, B, D).permute(1, 0, 2)  # (B, steps, D), steps=Tp-p

        # Map predictions to original space (aligned)
        if self.d == 0:
            y_out = yhat_d
            start = self.p
        else:
            # First predicted Δ^d corresponds to original time index (p + d)
            base_hist = x[:, : self.p + self.d, :]
            y_out = self._undifference_forecast(base_hist, yhat_d)
            start = self.p + self.d

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}

        if return_loss and y_true is not None:
            if y_true.shape != x.shape:
                raise ValueError(f"Expected y_true with shape {x.shape}, got {y_true.shape}")

            # Original-space MSE (aligned with y_out)
            y_true_aligned = y_true[:, start:, :]
            if y_true_aligned.shape[1] != y_out.shape[1]:
                m = min(y_true_aligned.shape[1], y_out.shape[1])
                y_true_aligned = y_true_aligned[:, :m, :]
                y_out_cmp = y_out[:, :m, :]
            else:
                y_out_cmp = y_out

            losses["mse"] = torch.mean((y_out_cmp - y_true_aligned) ** 2)
            losses["total"] = losses["mse"]

            # Differenced-space diagnostic MSE
            yd_true = self._difference(y_true)
            losses["mse_d"] = torch.mean((yhat_d - yd_true[:, self.p:, :]) ** 2)

        return ClassicalTSOutput(
            y=y_out,
            losses=losses,
            extras={
                "fitted": self._fitted,
                "space": "original",
                "y_differenced": yhat_d,
                "start_index_original": start,
            },
        )
