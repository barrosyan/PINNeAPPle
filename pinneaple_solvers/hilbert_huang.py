"""Hilbert-Huang transform (EMD + Hilbert) solver for time series."""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch

from .base import SolverBase, SolverOutput


def _find_extrema(x: torch.Tensor):
    """
    x: (T,)
    returns indices of local maxima and minima (1D)
    """
    dx = x[1:] - x[:-1]
    s = torch.sign(dx)
    # extrema where sign changes
    ds = s[1:] - s[:-1]
    # max: + -> -  (ds < 0)
    # min: - -> +  (ds > 0)
    max_idx = torch.where(ds < 0)[0] + 1
    min_idx = torch.where(ds > 0)[0] + 1
    return max_idx, min_idx


def _linear_envelope(x: torch.Tensor, idx: torch.Tensor):
    """
    Build linear envelope passing through points (idx, x[idx]).
    Includes endpoints for stability.
    """
    T = x.shape[0]
    if idx.numel() < 2:
        # fallback: flat envelope
        return torch.full_like(x, x.mean())

    # add endpoints
    idx2 = torch.cat([torch.tensor([0], device=x.device), idx, torch.tensor([T - 1], device=x.device)])
    val2 = x[idx2]

    t = torch.arange(T, device=x.device, dtype=x.dtype)
    env = torch.empty_like(x)

    for i in range(idx2.numel() - 1):
        a = idx2[i].item()
        b = idx2[i + 1].item()
        ta = t[a:b + 1]
        va = val2[i]
        vb = val2[i + 1]
        w = (ta - float(a)) / max(float(b - a), 1.0)
        env[a:b + 1] = (1 - w) * va + w * vb
    return env


def _is_imf(h: torch.Tensor):
    """
    IMF conditions (loose):
      - number of extrema and zero crossings differ by <= 1
      - local mean of envelopes is small-ish (handled in sifting loop)
    """
    T = h.shape[0]
    zc = torch.sum((h[:-1] * h[1:]) < 0).item()
    mx, mn = _find_extrema(h)
    ext = int(mx.numel() + mn.numel())
    return abs(ext - zc) <= 1


def _hilbert_analytic(x: torch.Tensor):
    """
    analytic signal via FFT method.
    x: (T,)
    returns z: complex (T,)
    """
    T = x.shape[0]
    X = torch.fft.fft(x)
    h = torch.zeros(T, device=x.device, dtype=X.dtype)

    if T % 2 == 0:
        h[0] = 1
        h[T // 2] = 1
        h[1:T // 2] = 2
    else:
        h[0] = 1
        h[1:(T + 1) // 2] = 2

    z = torch.fft.ifft(X * h)
    return z


class HilbertHuangSolver(SolverBase):
    """
    Hilbert–Huang Transform (HHT) MVP for 1D signals.

    Steps:
      1) EMD: decompose x into IMFs + residual
      2) Hilbert transform each IMF -> inst amplitude & frequency

    Notes:
      - This MVP targets research workflows; for production-grade EMD,
        you'd typically use a dedicated library.
    """
    def __init__(
        self,
        *,
        max_imfs: int = 8,
        max_sift: int = 50,
        stop_mean_tol: float = 1e-3,
        dt: float = 1.0,
    ):
        super().__init__()
        self.max_imfs = int(max_imfs)
        self.max_sift = int(max_sift)
        self.stop_mean_tol = float(stop_mean_tol)
        self.dt = float(dt)

    @torch.no_grad()
    def emd(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (T,) float
        returns:
          imfs: (K,T)
          residual: (T,)
        """
        r = x.clone()
        imfs = []

        for _ in range(self.max_imfs):
            h = r.clone()

            # if not enough extrema, stop
            mx, mn = _find_extrema(h)
            if (mx.numel() + mn.numel()) < 2:
                break

            for _s in range(self.max_sift):
                mx, mn = _find_extrema(h)
                if (mx.numel() + mn.numel()) < 2:
                    break
                upper = _linear_envelope(h, mx)
                lower = _linear_envelope(h, mn)
                m = 0.5 * (upper + lower)
                h = h - m
                if torch.mean(torch.abs(m)).item() < self.stop_mean_tol and _is_imf(h):
                    break

            imfs.append(h)
            r = r - h

        if len(imfs) == 0:
            imfs_t = torch.zeros((0, x.shape[0]), device=x.device, dtype=x.dtype)
        else:
            imfs_t = torch.stack(imfs, dim=0)
        return imfs_t, r

    def forward(self, x: torch.Tensor) -> SolverOutput:
        """
        x: (T,) or (B,T)
        Output:
          result: imfs (B,K,T) padded to max K across batch
          extras: residual, inst_amp, inst_freq (same padding)
        """
        if x.ndim == 1:
            xb = x[None, :]
        else:
            xb = x
        B, T = xb.shape
        device, dtype = xb.device, xb.dtype

        all_imfs = []
        all_res = []
        all_amp = []
        all_freq = []
        maxK = 0

        for b in range(B):
            imfs, res = self.emd(xb[b])
            K = imfs.shape[0]
            maxK = max(maxK, K)

            amps = []
            freqs = []
            for k in range(K):
                z = _hilbert_analytic(imfs[k])
                amp = torch.abs(z)
                phase = torch.unwrap(torch.angle(z))
                inst_freq = torch.zeros_like(phase)
                inst_freq[1:] = (phase[1:] - phase[:-1]) / (2.0 * torch.pi * self.dt)
                amps.append(amp)
                freqs.append(inst_freq)

            all_imfs.append(imfs)
            all_res.append(res)
            all_amp.append(torch.stack(amps, dim=0) if K > 0 else torch.zeros((0, T), device=device, dtype=dtype))
            all_freq.append(torch.stack(freqs, dim=0) if K > 0 else torch.zeros((0, T), device=device, dtype=dtype))

        # pad to (B,maxK,T)
        def pad_K(t: torch.Tensor, K: int):
            # t: (K0,T)
            if t.shape[0] == K:
                return t
            pad = torch.zeros((K - t.shape[0], T), device=device, dtype=dtype)
            return torch.cat([t, pad], dim=0)

        imfs_b = torch.stack([pad_K(t, maxK) for t in all_imfs], dim=0)  # (B,K,T)
        amp_b = torch.stack([pad_K(t, maxK) for t in all_amp], dim=0)
        frq_b = torch.stack([pad_K(t, maxK) for t in all_freq], dim=0)
        res_b = torch.stack(all_res, dim=0)  # (B,T)

        return SolverOutput(
            result=imfs_b if x.ndim == 2 else imfs_b[0],
            losses={"total": torch.tensor(0.0, device=device)},
            extras={
                "residual": res_b if x.ndim == 2 else res_b[0],
                "inst_amp": amp_b if x.ndim == 2 else amp_b[0],
                "inst_freq": frq_b if x.ndim == 2 else frq_b[0],
                "dt": self.dt,
            },
        )
