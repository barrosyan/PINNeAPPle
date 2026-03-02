"""CEEMDAN: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.

This implementation is an approximation built on top of the EMD/EEMD code in
``pinneaple_solvers.eemd``. It is suitable for feature extraction but is not a
drop-in replacement for specialized libraries.

Reference
---------
Torres et al. (2011) A complete ensemble empirical mode decomposition with
adaptive noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .eemd import emd_1d
from .registry import SolverRegistry


def ceemdan_1d(
    x: np.ndarray,
    *,
    n_ensembles: int = 50,
    noise_std: float = 0.2,
    max_imfs: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Approximate CEEMDAN. Returns imfs (K,T)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    s = np.std(x) + 1e-12

    residue = x.copy()
    imfs = []

    for k in range(int(max_imfs)):
        # ensemble mean of first IMF from noisy residue
        imf_acc = np.zeros(T, dtype=float)
        for _ in range(int(n_ensembles)):
            wn = rng.normal(0.0, float(noise_std) * s, size=T)
            cur = residue + wn
            cur_imfs = emd_1d(cur, max_imfs=1)
            if cur_imfs:
                imf_acc += cur_imfs[0]
        imf_k = imf_acc / float(max(1, n_ensembles))
        imfs.append(imf_k)
        residue = residue - imf_k

        # stop if residue becomes monotonic-ish
        if len(emd_1d(residue, max_imfs=1)) == 0:
            break

    return np.stack(imfs, axis=0).astype(np.float32) if imfs else np.zeros((0, T), dtype=np.float32)


@SolverRegistry.register(
    name="ceemdan",
    family="hilbert_huang",
    description="CEEMDAN (approx): Complete Ensemble EMD with Adaptive Noise.",
    tags=["time_series", "decomposition", "hht"],
)
class CEEMDANSolver(SolverBase):
    def __init__(
        self,
        n_ensembles: int = 50,
        noise_std: float = 0.2,
        max_imfs: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_ensembles = int(n_ensembles)
        self.noise_std = float(noise_std)
        self.max_imfs = int(max_imfs)
        self.seed = seed

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        outs = []
        for b in range(B):
            imfs = ceemdan_1d(
                xt[b],
                n_ensembles=self.n_ensembles,
                noise_std=self.noise_std,
                max_imfs=self.max_imfs,
                seed=None if self.seed is None else int(self.seed) + b,
            )
            outs.append(imfs)
        K = max(o.shape[0] for o in outs) if outs else 0
        out = np.zeros((B, K, T), dtype=np.float32)
        for b, o in enumerate(outs):
            out[b, : o.shape[0], :] = o
        return SolverOutput(result=torch.from_numpy(out).to(x.device), losses={}, extras={"K": int(K)})
