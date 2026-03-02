"""Wavelet-based decomposition/denoising features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import pywt

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def wavedec_features(x: np.ndarray, wavelet: str = "db4", level: int = 4, mode: str = "periodization") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return concatenated detail coefficients as features."""
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=int(level), mode=mode)
    # coeffs[0]=approx, others=details
    details = coeffs[1:]
    # pad details to same length by interpolation (simple)
    T = len(x)
    feats = []
    for d in details:
        if len(d) == T:
            feats.append(d)
        else:
            # linear resample
            idx = np.linspace(0, 1, num=len(d))
            idx2 = np.linspace(0, 1, num=T)
            feats.append(np.interp(idx2, idx, d))
    F = np.stack(feats, axis=-1).astype(np.float32) if feats else np.zeros((T, 0), dtype=np.float32)
    meta = {"wavelet": wavelet, "level": int(level)}
    return F, meta


@SolverRegistry.register(
    name="wavelet",
    family="time_frequency",
    description="Wavelet decomposition features (detail coefficients).",
    tags=["time_series", "time_frequency"],
)
class WaveletSolver(SolverBase):
    def __init__(self, wavelet: str = "db4", level: int = 4, mode: str = "periodization"):
        super().__init__()
        self.wavelet = str(wavelet)
        self.level = int(level)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        feats = []
        meta0 = None
        for b in range(B):
            f, meta = wavedec_features(xt[b], wavelet=self.wavelet, level=self.level, mode=self.mode)
            feats.append(f)
            meta0 = meta
        result = torch.from_numpy(np.stack(feats, axis=0)).to(x.device)  # (B,T,F)
        return SolverOutput(result=result, losses={}, extras={"meta": meta0 or {}})
