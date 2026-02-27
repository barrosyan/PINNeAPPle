"""
pinneaple_solvers: Hilbert–Huang Transform (EMD + Hilbert) demo

What this shows
--------------
- How to decompose a non-stationary 1D signal into IMFs + residual.
- How to obtain instantaneous amplitude and frequency (per IMF).
- How to validate reconstruction quality (sum(IMFs) + residual ~= signal).

Run
---
python examples/pinneaple_solvers/02_solvers_hilbert_huang_decompose.py

Notes
-----
This is an MVP EMD implementation focused on research workflows.
For heavy-duty production EMD, you'd typically switch to a dedicated library.
"""

from __future__ import annotations

import math

import torch

from pinneaple_solvers.hilbert_huang import HilbertHuangSolver


def make_signal(T: int = 2048, dt: float = 1.0 / 200.0) -> torch.Tensor:
    """Non-stationary signal: AM-FM + trend + noise."""
    t = torch.arange(T) * dt
    # AM-FM component: frequency slowly increases
    f0, f1 = 3.0, 18.0
    inst_f = f0 + (f1 - f0) * (t / t.max())
    phase = 2 * math.pi * torch.cumsum(inst_f * dt, dim=0)
    am = 0.6 + 0.4 * torch.sin(2 * math.pi * 0.25 * t)
    x1 = am * torch.sin(phase)

    # low-frequency oscillation
    x2 = 0.35 * torch.sin(2 * math.pi * 1.0 * t + 0.4)

    # slow trend
    trend = 0.2 * (t - t.mean())

    # light noise
    noise = 0.02 * torch.randn_like(t)
    return x1 + x2 + trend + noise


def main():
    torch.manual_seed(0)

    T = 2048
    dt = 1.0 / 200.0
    x = make_signal(T=T, dt=dt)

    solver = HilbertHuangSolver(max_imfs=8, max_sift=60, stop_mean_tol=2e-3, dt=dt)
    out = solver(x)

    imfs = out.result  # (K,T)
    res = out.extras["residual"]
    amp = out.extras["inst_amp"]
    frq = out.extras["inst_freq"]

    K = imfs.shape[0]
    x_rec = imfs.sum(dim=0) + res
    rmse = torch.sqrt(torch.mean((x - x_rec) ** 2)).item()

    print("--- Hilbert–Huang demo")
    print(f"signal: T={T}, dt={dt:.6f}s")
    print(f"imfs:   K={K}")
    print(f"recon rmse: {rmse:.6e}")

    # Simple summary: median instantaneous frequency per IMF
    # (skip zeros from padding / edges)
    for k in range(K):
        fk = frq[k]
        valid = torch.isfinite(fk) & (fk.abs() > 1e-8)
        med = fk[valid].median().item() if valid.any() else float("nan")
        a_med = amp[k][torch.isfinite(amp[k])].median().item()
        print(f"IMF{k:02d}: median_f={med:8.3f} Hz | median_amp={a_med:8.4f}")


if __name__ == "__main__":
    main()