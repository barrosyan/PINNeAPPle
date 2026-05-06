from __future__ import annotations
"""ESN-RC hybrid for reservoir computing with readout regression."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import RCBase, RCOutput
from .esn import EchoStateNetwork


class ESNRC(RCBase):
    """
    ESN-RC (Reservoir Computing) variant:
      - Uses ESN reservoir but augments readout with nonlinear transforms
        (e.g., h, h^2, h^3) to approximate richer dynamics while keeping
        closed-form ridge training.

    This is a common “physics RC trick” when dynamics are polynomial-like.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        reservoir_dim: int = 1024,
        *,
        spectral_radius: float = 0.9,
        leak: float = 1.0,
        input_scale: float = 1.0,
        l2: float = 1e-6,
        poly_degree: int = 2,   # include h^2..h^k
        use_skip: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.esn = EchoStateNetwork(
            in_dim=in_dim,
            out_dim=out_dim,
            reservoir_dim=reservoir_dim,
            spectral_radius=spectral_radius,
            leak=leak,
            input_scale=input_scale,
            l2=l2,
            use_skip=False,  # we'll manage features ourselves
            use_bias=False,
        )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.reservoir_dim = int(reservoir_dim)
        self.poly_degree = int(poly_degree)
        self.use_skip = bool(use_skip)
        self.use_bias = bool(use_bias)
        self.l2 = float(l2)

        feat_dim = reservoir_dim * self.poly_degree + (in_dim if self.use_skip else 0) + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(feat_dim, out_dim), requires_grad=False)
        self._fitted = False

    def _features(self, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        feats = []
        # h, h^2, ..., h^k
        p = h
        feats.append(p)
        for _ in range(2, self.poly_degree + 1):
            p = p * h
            feats.append(p)
        if self.use_skip:
            feats.append(x_t)
        if self.use_bias:
            feats.append(torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, *, washout: int = 0) -> "ESNRC":
        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        X_list, Y_list = [], []
        for t in range(T):
            h = self.esn._step(x[:, t, :], h)
            if t >= int(washout):
                X_list.append(self._features(h, x[:, t, :]))
                Y_list.append(y[:, t, :])

        Xr = torch.cat(X_list, dim=0)
        Yr = torch.cat(Y_list, dim=0)
        W = self.ridge_solve(Xr, Yr, l2=self.l2)
        self.W_out.copy_(W)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor, *, y_true: Optional[torch.Tensor] = None, return_loss: bool = False, washout: int = 0) -> RCOutput:
        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        ys = []
        for t in range(T):
            h = self.esn._step(x[:, t, :], h)
            y_t = self._features(h, x[:, t, :]) @ self.W_out
            ys.append(y_t)

        y_hat = torch.stack(ys, dim=1)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if washout > 0:
                losses["mse"] = torch.mean((y_hat[:, washout:, :] - y_true[:, washout:, :]) ** 2)
            else:
                losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})
