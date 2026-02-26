from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn


def pinball_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor, q: float) -> torch.Tensor:
    q = float(q)
    e = y_true - y_pred
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


@dataclass
class QuantileConfig:
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)


class QuantileHead(nn.Module):
    """Wrap a point forecaster and project outputs to quantiles.

    Assumes base(x) returns (B,H) or (B,H,1). Output: (B,H,Q).
    """

    def __init__(self, base: nn.Module, cfg: QuantileConfig, hidden_dim: int = 64):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(cfg.quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_point = self.base(x)
        if y_point.ndim == 2:
            y_point = y_point.unsqueeze(-1)  # (B,H,1)
        if y_point.shape[-1] != 1:
            raise ValueError("QuantileHead expects single-target base output (C=1).")
        B, H, _ = y_point.shape
        yq = self.proj(y_point.reshape(B * H, 1)).reshape(B, H, -1)
        return yq


class QuantileLoss:
    def __init__(self, quantiles: Sequence[float]):
        self.quantiles = list(quantiles)

    def __call__(self, model: nn.Module, y_pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y_true = batch["y"]
        if y_true.ndim == 3:
            if y_true.shape[-1] != 1:
                raise ValueError("QuantileLoss expects single-target y with C=1.")
            y_true = y_true[..., 0]  # (B,H)
        elif y_true.ndim != 2:
            raise ValueError(f"Unexpected y shape: {tuple(y_true.shape)}")

        if y_pred.ndim != 3:
            raise ValueError(f"Expected y_pred (B,H,Q), got {tuple(y_pred.shape)}")

        total = 0.0
        for qi, q in enumerate(self.quantiles):
            total = total + pinball_loss_torch(y_pred[:, :, qi], y_true, q)
        total = total / max(1, len(self.quantiles))
        return {"total": total}