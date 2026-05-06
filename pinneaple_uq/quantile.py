"""Quantile (distribution-free) uncertainty estimation.

Provides pinball-loss training for quantile regression models and a general
QuantileHead that wraps any point-prediction model.

Unlike conformal prediction, quantile regression *learns* conditional quantiles
from data rather than relying on post-hoc calibration.  Use
:class:`~pinneaple_uq.conformal.ConformalPredictor` when distribution-free
coverage guarantees are required.

Quick start::

    from pinneaple_uq import QuantileHead, QuantileLoss, QuantileConfig

    cfg  = QuantileConfig(quantiles=(0.1, 0.5, 0.9))
    head = QuantileHead(base_model, cfg)
    loss = QuantileLoss(cfg.quantiles)

    for x_batch, y_batch in loader:
        y_pred = head(x_batch)          # (B, H, Q)
        metrics = loss(head, y_pred, {"y": y_batch})
        metrics["total"].backward()
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn


def pinball_loss_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    q: float,
) -> torch.Tensor:
    """Pinball (quantile) loss for a single quantile level *q*.

    Parameters
    ----------
    y_pred : Tensor — predicted quantile, shape ``(B, H)`` or ``(N,)``.
    y_true : Tensor — targets, same shape as *y_pred*.
    q : float — quantile level in ``(0, 1)``.

    Returns
    -------
    Tensor — scalar mean pinball loss.
    """
    q = float(q)
    e = y_true - y_pred
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


@dataclass
class QuantileConfig:
    """Configuration for quantile prediction.

    Parameters
    ----------
    quantiles : sequence of floats in ``(0, 1)``.
        Default ``(0.1, 0.5, 0.9)`` gives 80 % prediction interval + median.
    """
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)


class QuantileHead(nn.Module):
    """Wrap a point-prediction model and project outputs to multiple quantiles.

    Assumes ``base(x)`` returns ``(B, H)`` or ``(B, H, 1)``.
    Output shape: ``(B, H, Q)`` where *Q* = ``len(cfg.quantiles)``.

    Parameters
    ----------
    base : nn.Module — backbone forecaster (any architecture).
    cfg : QuantileConfig — quantile levels.
    hidden_dim : int — width of the projection head.
    """

    def __init__(
        self,
        base: nn.Module,
        cfg: QuantileConfig,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(cfg.quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantile predictions, shape ``(B, H, Q)``."""
        y_point = self.base(x)
        if y_point.ndim == 2:
            y_point = y_point.unsqueeze(-1)  # (B, H, 1)
        if y_point.shape[-1] != 1:
            raise ValueError("QuantileHead expects single-target base output (C=1).")
        B, H, _ = y_point.shape
        return self.proj(y_point.reshape(B * H, 1)).reshape(B, H, -1)


class QuantileLoss:
    """Multi-quantile pinball loss compatible with the Pinneaple Trainer.

    Parameters
    ----------
    quantiles : sequence of floats.
    """

    def __init__(self, quantiles: Sequence[float]) -> None:
        self.quantiles = list(quantiles)

    def __call__(
        self,
        model: nn.Module,
        y_pred: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute averaged pinball loss across all quantile levels.

        Parameters
        ----------
        model : nn.Module — unused (required for Trainer compatibility).
        y_pred : Tensor — ``(B, H, Q)``.
        batch : dict — must contain key ``"y"`` with target ``(B, H)`` or ``(B, H, 1)``.

        Returns
        -------
        dict with ``"total"`` key for Trainer.
        """
        y_true = batch["y"]
        if y_true.ndim == 3:
            if y_true.shape[-1] != 1:
                raise ValueError("QuantileLoss expects single-target y with C=1.")
            y_true = y_true[..., 0]  # (B, H)
        elif y_true.ndim != 2:
            raise ValueError(f"Unexpected y shape: {tuple(y_true.shape)}")

        if y_pred.ndim != 3:
            raise ValueError(f"Expected y_pred (B, H, Q), got {tuple(y_pred.shape)}")

        total: torch.Tensor = sum(  # type: ignore[assignment]
            pinball_loss_torch(y_pred[:, :, qi], y_true, q)
            for qi, q in enumerate(self.quantiles)
        )
        total = total / max(1, len(self.quantiles))
        return {"total": total}
