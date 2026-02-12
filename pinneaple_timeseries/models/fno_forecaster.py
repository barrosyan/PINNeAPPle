from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from pinneaple_models.neural_operators.fno import FourierNeuralOperator


@dataclass
class FNOForecastConfig:
    input_len: int
    horizon: int
    n_features: int
    n_targets: int

    width: int = 64
    modes: int = 16
    layers: int = 4


class FNOForecaster(nn.Module):
    """
    Forecaster TS padrão usando FNO-1D.
    Convenção:
      x: (B, L_in, F)
      y_hat: (B, H, C)
    """

    def __init__(self, cfg: FNOForecastConfig):
        super().__init__()
        self.cfg = cfg

        self.fno = FourierNeuralOperator(
            in_channels=cfg.n_features,
            out_channels=cfg.n_targets,
            width=cfg.width,
            modes=cfg.modes,
            layers=cfg.layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_in, F) -> (B, F, L_in)
        if x.ndim != 3:
            raise ValueError("x deve ser (B, L_in, F)")

        B, L_in, F = x.shape
        H = self.cfg.horizon
        C = self.cfg.n_targets

        if L_in != self.cfg.input_len:
            # não forço, mas aviso no erro pra padronizar
            raise ValueError(f"Esperado L_in={self.cfg.input_len}, veio {L_in}")

        u = x.transpose(1, 2).contiguous()  # (B, F, L_in)

        # pad futuro com zeros: (B, F, L_in+H)
        u_pad = torch.zeros(B, F, L_in + H, device=u.device, dtype=u.dtype)
        u_pad[:, :, :L_in] = u

        # FNO retorna OperatorOutput(y=...) no seu código
        out = self.fno(u_pad, y_true=None, return_loss=False)
        y_full = out.y  # (B, C, L_in+H)

        # pega apenas horizonte futuro
        y_fut = y_full[:, :, -H:]  # (B, C, H)

        # volta p/ (B, H, C)
        y_hat = y_fut.transpose(1, 2).contiguous()
        return y_hat
