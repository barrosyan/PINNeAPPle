from __future__ import annotations
"""Echo state network with reservoir dynamics."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


def _spectral_radius(W: torch.Tensor, iters: int = 50) -> torch.Tensor:
    # power iteration approximation of largest eigenvalue magnitude
    v = torch.randn(W.shape[0], 1, device=W.device, dtype=W.dtype)
    for _ in range(iters):
        v = W @ v
        v = v / (v.norm() + 1e-12)
    lam = (v.t() @ (W @ v)).squeeze()
    return lam.abs().clamp_min(1e-12)


class EchoStateNetwork(RCBase):
    """
    Echo State Network (ESN):
      h_t = (1-leak)*h_{t-1} + leak*tanh(W_in x_t + W h_{t-1})
      y_t = [h_t, x_t, 1] W_out  (trained by ridge)

    Input:
      x: (B,T,in_dim)
    Output:
      y: (B,T,out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        reservoir_dim: int = 1024,
        spectral_radius: float = 0.9,
        leak: float = 1.0,
        input_scale: float = 1.0,
        l2: float = 1e-6,
        use_skip: bool = True,  # include x in readout
        use_bias: bool = True,
        freeze_random: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.reservoir_dim = int(reservoir_dim)
        self.spectral_radius = float(spectral_radius)
        self.leak = float(leak)
        self.input_scale = float(input_scale)
        self.l2 = float(l2)
        self.use_skip = bool(use_skip)
        self.use_bias = bool(use_bias)

        Win = torch.randn(self.in_dim, self.reservoir_dim) * (self.input_scale / max(self.in_dim, 1) ** 0.5)
        W = torch.randn(self.reservoir_dim, self.reservoir_dim) * (1.0 / max(self.reservoir_dim, 1) ** 0.5)

        # scale to desired spectral radius
        with torch.no_grad():
            sr = _spectral_radius(W.float()).to(W.dtype)
            W = W * (self.spectral_radius / sr)

        self.W_in = nn.Parameter(Win)
        self.W = nn.Parameter(W)

        if freeze_random:
            self.W_in.requires_grad_(False)
            self.W.requires_grad_(False)

        readout_dim = self.reservoir_dim + (self.in_dim if self.use_skip else 0) + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(readout_dim, self.out_dim), requires_grad=False)

        self._fitted = False

    def _step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre = x_t @ self.W_in + h @ self.W
        h_tilde = torch.tanh(pre)
        return (1.0 - self.leak) * h + self.leak * h_tilde

    def _readout_features(self, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        feats = [h]
        if self.use_skip:
            feats.append(x_t)
        if self.use_bias:
            feats.append(torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
        return torch.cat(feats, dim=-1)

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,  # (B,T,in_dim)
        y: torch.Tensor,  # (B,T,out_dim)
        *,
        washout: int = 0,
    ) -> "EchoStateNetwork":
        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        X_list, Y_list = [], []
        for t in range(T):
            h = self._step(x[:, t, :], h)
            if t >= int(washout):
                X_list.append(self._readout_features(h, x[:, t, :]))
                Y_list.append(y[:, t, :])

        Xr = torch.cat(X_list, dim=0)  # (B*(T-w), F)
        Yr = torch.cat(Y_list, dim=0)  # (B*(T-w), O)
        W_out = self.ridge_solve(Xr, Yr, l2=self.l2)
        self.W_out.copy_(W_out)
        self._fitted = True
        return self

    def forward(
        self,
        x: torch.Tensor,  # (B,T,in_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        washout: int = 0,
    ) -> RCOutput:
        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        ys = []
        for t in range(T):
            h = self._step(x[:, t, :], h)
            feats = self._readout_features(h, x[:, t, :])
            y_t = feats @ self.W_out
            ys.append(y_t)

        y_hat = torch.stack(ys, dim=1)  # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if washout > 0:
                losses["mse"] = torch.mean((y_hat[:, washout:, :] - y_true[:, washout:, :]) ** 2)
            else:
                losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})
