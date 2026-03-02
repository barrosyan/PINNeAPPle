"""SST: Synchrosqueezed transform (MVP).

Full SST is involved; this MVP implements a simplified STFT-based frequency
reassignment (synchrosqueezing-like) using phase derivatives.

For feature extraction, this provides a sharper time-frequency map than raw
STFT in many cases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from scipy.signal import stft

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def sst_stft_1d(
    x: np.ndarray,
    *,
    fs: float = 1.0,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (SST_like, freqs, times)."""
    f, t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None)
    # instantaneous frequency estimate from phase derivative along time
    phase = np.unwrap(np.angle(Z), axis=1)
    dphi = np.diff(phase, axis=1)
    dt = np.diff(t)
    dt = dt[None, :] + eps
    inst_freq = dphi / (2 * np.pi * dt)  # cycles per second

    # squeeze: move each time-frequency bin energy to nearest inst_freq
    S = np.zeros_like(Z, dtype=np.complex128)
    mag = np.abs(Z)
    for ti in range(inst_freq.shape[1]):
        w = inst_freq[:, ti]
        # map to index
        idx = np.clip(np.round(np.interp(w, f, np.arange(len(f)))).astype(int), 0, len(f) - 1)
        for fi, jj in enumerate(idx):
            if mag[fi, ti] > eps:
                S[jj, ti] += Z[fi, ti]
    return S.astype(np.complex64), f.astype(np.float32), t.astype(np.float32)


@SolverRegistry.register(
    name="sst",
    family="time_frequency",
    description="Synchrosqueezed transform (STFT-based MVP).",
    tags=["time_series", "time_frequency"],
)
class SSTSolver(SolverBase):
    def __init__(
        self,
        fs: float = 1.0,
        nperseg: int = 128,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
    ):
        super().__init__()
        self.fs = float(fs)
        self.nperseg = int(nperseg)
        self.noverlap = None if noverlap is None else int(noverlap)
        self.nfft = None if nfft is None else int(nfft)

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        maps = []
        freqs = None
        times = None
        for b in range(B):
            S, f, t = sst_stft_1d(xt[b], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
            maps.append(np.abs(S).astype(np.float32))  # magnitude TF map
            freqs = f
            times = t
        result = torch.from_numpy(np.stack(maps, axis=0)).to(x.device)  # (B,F,Tframes)
        return SolverOutput(result=result, losses={}, extras={"freqs": freqs, "times": times})
