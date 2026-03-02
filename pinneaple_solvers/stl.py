"""STL: Seasonal-Trend decomposition using Loess."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from statsmodels.tsa.seasonal import STL

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


@SolverRegistry.register(
    name="stl",
    family="decomposition",
    description="STL decomposition into trend/seasonal/resid (statsmodels).",
    tags=["time_series", "decomposition"],
)
class STLSolver(SolverBase):
    def __init__(self, period: int = 24, robust: bool = True):
        super().__init__()
        self.period = int(period)
        self.robust = bool(robust)

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        parts = []
        for b in range(B):
            res = STL(xt[b], period=self.period, robust=self.robust).fit()
            # (T,3): trend, seasonal, resid
            parts.append(np.stack([res.trend, res.seasonal, res.resid], axis=-1).astype(np.float32))
        result = torch.from_numpy(np.stack(parts, axis=0)).to(x.device)  # (B,T,3)
        return SolverOutput(result=result, losses={}, extras={"period": self.period})
