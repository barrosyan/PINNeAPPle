from __future__ import annotations
"""1D convolutional model for sequence data (forecasting, denoising, operator learning, PDE surrogate)."""

from dataclasses import dataclass
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ConvModelBase, ConvOutput

Mode = Literal["causal_forecast", "denoise", "operator", "pde_surrogate"]
NormType = Literal["none", "batch", "group", "layer"]
DropoutType = Literal["standard", "channel"]
DilationSchedule = Literal["constant", "exponential"]


def _make_norm(norm: NormType, channels: int) -> nn.Module:
    if norm == "none":
        return nn.Identity()
    if norm == "batch":
        return nn.BatchNorm1d(channels)
    if norm == "group":
        # common practical default: up to 8 groups, but must divide channels
        # fallback to 1 group (LayerNorm-like) if not divisible
        for g in (8, 4, 2):
            if channels % g == 0:
                return nn.GroupNorm(g, channels)
        return nn.GroupNorm(1, channels)
    if norm == "layer":
        # LayerNorm over channel dim -> need (B, L, C)
        return nn.LayerNorm(channels)
    raise ValueError(f"Unknown norm: {norm}")


class _LayerNorm1d(nn.Module):
    """Applies LayerNorm over channel dim for (B, C, L) tensors."""
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> (B, L, C) -> LN -> (B, C, L)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _make_norm1d(norm: NormType, channels: int) -> nn.Module:
    if norm == "layer":
        return _LayerNorm1d(channels)
    return _make_norm(norm, channels)


def _make_dropout(dropout: float, kind: DropoutType) -> nn.Module:
    p = float(dropout)
    if p <= 0.0:
        return nn.Identity()
    if kind == "standard":
        return nn.Dropout(p)
    if kind == "channel":
        # channel-wise dropout for Conv1d (zeros entire channels across L)
        return nn.Dropout1d(p)
    raise ValueError(f"Unknown dropout kind: {kind}")


class _ConvBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        causal: bool,
        norm: NormType,
        dropout: float,
        dropout_kind: DropoutType,
        activation: Literal["gelu", "silu", "relu"] = "gelu",
    ):
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)

        if k <= 0 or d <= 0:
            raise ValueError("kernel_size and dilation must be positive integers.")

        # Padding strategy
        if causal:
            # left padding only to preserve length while preventing future leakage
            self.left_pad = (k - 1) * d
            pad = 0
        else:
            # "same" length requires odd kernel for exact symmetry
            # pad computed to preserve length when stride=1
            pad = ((k - 1) * d) // 2
            self.left_pad = 0

        self.causal = bool(causal)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=pad)
        self.norm1 = _make_norm1d(norm, channels)
        self.drop = _make_dropout(dropout, dropout_kind)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=pad)
        self.norm2 = _make_norm1d(norm, channels)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal and self.left_pad > 0:
            x = nn.functional.pad(x, (self.left_pad, 0))  # pad L dimension (left, right)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        if self.causal and self.left_pad > 0:
            h = nn.functional.pad(h, (self.left_pad, 0))

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h


class Conv1DModel(ConvModelBase):
    """
    Flexible 1D Conv model supporting:
      - causal forecasting
      - denoising
      - operator learning
      - PDE 1D surrogate

    Inputs:
      x:    (B, C_in, L)
      cond: (B, C_cond, L) optional conditioning (coords, parameters, forcing, etc.)

    Outputs:
      - default: y = (B, C_out, L)
      - forecasting with forecast_horizon>0: y = (B, C_out, H) taking the last H steps
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: Mode = "denoise",
        cond_channels: int = 0,  # used mainly for operator/pde modes; can be used in any mode
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        dropout_kind: DropoutType = "channel",
        residual: bool = True,
        residual_scale: float = 1.0,
        learnable_residual_scale: bool = False,
        residual_scale_init: float = 0.1,
        norm: NormType = "group",
        activation: Literal["gelu", "silu", "relu"] = "gelu",
        dilation_schedule: DilationSchedule = "constant",
        dilation_base: int = 1,
        causal: Optional[bool] = None,  # if None: inferred from mode
        forecast_horizon: int = 0,       # if > 0, returns last H steps (useful for forecasting)
        enforce_odd_kernel: bool = True, # recommended to ensure true "same-length" when non-causal
    ):
        super().__init__()

        self.mode: Mode = mode
        inferred_causal = (mode == "causal_forecast")
        self.causal = bool(inferred_causal if causal is None else causal)

        k = int(kernel_size)
        if enforce_odd_kernel and (not self.causal) and (k % 2 == 0):
            raise ValueError("Non-causal 'same-length' conv requires odd kernel_size. Set enforce_odd_kernel=False to override.")

        self.forecast_horizon = int(forecast_horizon)
        if self.forecast_horizon < 0:
            raise ValueError("forecast_horizon must be >= 0.")

        in_total = int(in_channels) + int(cond_channels)
        self.in_proj = nn.Conv1d(in_total, int(hidden_channels), kernel_size=1)

        self.blocks = nn.ModuleList()
        self.residual = bool(residual)

        # residual scaling (helps stability as depth grows)
        if self.residual:
            if learnable_residual_scale:
                self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
            else:
                self.register_buffer("res_scale", torch.tensor(float(residual_scale)), persistent=False)
        else:
            self.register_buffer("res_scale", torch.tensor(0.0), persistent=False)

        nb = int(num_blocks)
        if nb <= 0:
            raise ValueError("num_blocks must be >= 1.")

        base = int(dilation_base)
        if base <= 0:
            raise ValueError("dilation_base must be >= 1.")

        for i in range(nb):
            if dilation_schedule == "constant":
                d = base
            elif dilation_schedule == "exponential":
                d = base * (2 ** i)
            else:
                raise ValueError(f"Unknown dilation_schedule: {dilation_schedule}")

            self.blocks.append(
                _ConvBlock1D(
                    int(hidden_channels),
                    kernel_size=k,
                    dilation=d,
                    causal=self.causal,
                    norm=norm,
                    dropout=float(dropout),
                    dropout_kind=dropout_kind,
                    activation=activation,
                )
            )

        self.out_proj = nn.Conv1d(int(hidden_channels), int(out_channels), kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> ConvOutput:
        if cond is not None:
            # operator/PDE surrogate typically passes coords/params/forcing here
            x = torch.cat([x, cond], dim=1)

        h = self.in_proj(x)

        for blk in self.blocks:
            h2 = blk(h)
            if self.residual:
                h = h + self.res_scale * h2
            else:
                h = h2

        y = self.out_proj(h)

        # Forecasting convenience: return last H steps only
        if self.mode == "causal_forecast" and self.forecast_horizon > 0:
            y = y[..., -self.forecast_horizon :]

        losses: Dict[str, torch.Tensor] = {}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return ConvOutput(y=y, losses=losses, extras={"mode": self.mode, "causal": self.causal})