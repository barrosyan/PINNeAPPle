"""SSA: Singular Spectrum Analysis for 1D time series."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def ssa_decompose(x: np.ndarray, L: int = 64, r: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reconstructed components (r,T), singular values, and U.

    L: window length
    r: number of components
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    L = int(min(max(2, L), T - 1))
    K = T - L + 1
    # trajectory matrix
    X = np.column_stack([x[i : i + L] for i in range(K)])  # (L,K)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(min(r, U.shape[1]))

    comps = []
    for i in range(r):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        # diagonal averaging (Hankelization)
        recon = np.zeros(T)
        counts = np.zeros(T)
        for a in range(L):
            for b in range(K):
                recon[a + b] += Xi[a, b]
                counts[a + b] += 1
        recon = recon / np.maximum(counts, 1)
        comps.append(recon)

    return np.stack(comps, axis=0).astype(np.float32), s[:r].astype(np.float32), U[:, :r].astype(np.float32)


@SolverRegistry.register(
    name="ssa",
    family="decomposition",
    description="Singular Spectrum Analysis (SSA) decomposition components.",
    tags=["time_series", "decomposition"],
)
class SSASolver(SolverBase):
    def __init__(self, L: int = 64, r: int = 10):
        super().__init__()
        self.L = int(L)
        self.r = int(r)

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        comps_list = []
        svals_list = []
        for b in range(B):
            comps, s, _U = ssa_decompose(xt[b], L=self.L, r=self.r)
            comps_list.append(comps)
            svals_list.append(s)
        R = max(c.shape[0] for c in comps_list) if comps_list else 0
        out = np.zeros((B, R, T), dtype=np.float32)
        for b, c in enumerate(comps_list):
            out[b, : c.shape[0], :] = c
        return SolverOutput(result=torch.from_numpy(out).to(x.device), losses={}, extras={"svals": svals_list})
