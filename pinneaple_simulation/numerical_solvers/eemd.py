"""EEMD: Ensemble Empirical Mode Decomposition.

This is a lightweight implementation intended for feature extraction /
preprocessing. It trades some fidelity for simplicity and few dependencies.

References
----------
Wu & Huang (2009) Ensemble empirical mode decomposition: a noise-assisted data
analysis method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from scipy.interpolate import CubicSpline

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def _extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of local maxima and minima."""
    dx = np.diff(x)
    # sign changes
    s = np.sign(dx)
    ds = np.diff(s)
    max_idx = np.where(ds < 0)[0] + 1
    min_idx = np.where(ds > 0)[0] + 1
    return max_idx, min_idx


def _envelope(t: np.ndarray, x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size < 2:
        # fallback: constant envelope
        return np.full_like(x, float(np.mean(x)))
    # include endpoints for stability
    pts = np.unique(np.concatenate([[0], idx, [len(x) - 1]]))
    cs = CubicSpline(t[pts], x[pts], bc_type="natural")
    return cs(t)


def emd_1d(
    x: np.ndarray,
    *,
    max_imfs: int = 10,
    max_sift: int = 50,
    stop_thresh: float = 0.05,
) -> List[np.ndarray]:
    """Very small EMD implementation for 1D signals."""
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float)
    residue = x.copy()
    imfs: List[np.ndarray] = []

    for _k in range(int(max_imfs)):
        h = residue.copy()
        for _ in range(int(max_sift)):
            max_i, min_i = _extrema(h)
            if max_i.size + min_i.size < 4:
                break
            upper = _envelope(t, h, max_i)
            lower = _envelope(t, h, min_i)
            m = 0.5 * (upper + lower)
            h_new = h - m
            # stopping: mean envelope small relative to signal
            denom = np.maximum(1e-12, np.mean(np.abs(h)))
            if np.mean(np.abs(m)) / denom < float(stop_thresh):
                h = h_new
                break
            h = h_new

        imfs.append(h)
        residue = residue - h

        # stop if residue is monotonic
        max_i, min_i = _extrema(residue)
        if max_i.size + min_i.size < 4:
            break

    return imfs


def eemd_1d(
    x: np.ndarray,
    *,
    n_ensembles: int = 50,
    noise_std: float = 0.2,
    max_imfs: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    """EEMD for 1D: returns imfs array (K, T)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    T = x.shape[0]

    imf_accum: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None

    s = np.std(x) + 1e-12
    for _ in range(int(n_ensembles)):
        xn = x + rng.normal(0.0, float(noise_std) * s, size=T)
        imfs = emd_1d(xn, max_imfs=max_imfs)
        K = len(imfs)
        if imf_accum is None:
            imf_accum = np.zeros((max_imfs, T), dtype=float)
            counts = np.zeros((max_imfs,), dtype=float)
        for k in range(min(K, max_imfs)):
            imf_accum[k] += imfs[k]
            counts[k] += 1.0

    assert imf_accum is not None and counts is not None
    counts = np.maximum(counts, 1.0)
    imf_accum = imf_accum / counts[:, None]
    # trim unused
    used = int(np.max(np.where(counts > 0)[0]) + 1) if np.any(counts > 0) else 0
    return imf_accum[:used]


@SolverRegistry.register(
    name="eemd",
    family="hilbert_huang",
    description="Ensemble Empirical Mode Decomposition (noise-assisted EMD).",
    tags=["time_series", "decomposition", "hht"],
)
class EEMDSolver(SolverBase):
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
        """x: (B,T) or (T,) -> imfs (B,K,T)."""
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        imfs_list = []
        for b in range(B):
            imfs = eemd_1d(
                xt[b],
                n_ensembles=self.n_ensembles,
                noise_std=self.noise_std,
                max_imfs=self.max_imfs,
                seed=None if self.seed is None else int(self.seed) + b,
            )
            imfs_list.append(imfs)
        K = max(im.shape[0] for im in imfs_list) if imfs_list else 0
        out = np.zeros((B, K, T), dtype=np.float32)
        for b, im in enumerate(imfs_list):
            out[b, : im.shape[0], :] = im.astype(np.float32)

        result = torch.from_numpy(out).to(x.device)
        return SolverOutput(result=result, losses={}, extras={"K": int(K)})
