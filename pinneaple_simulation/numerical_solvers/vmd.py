"""VMD: Variational Mode Decomposition.

Reference
---------
Dragomiretskiy & Zosso (2014) Variational Mode Decomposition.

This implementation follows the standard ADMM updates in the frequency domain.
It is designed for 1D signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def vmd_1d(
    x: np.ndarray,
    *,
    K: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    DC: bool = False,
    init: str = "uniform",  # uniform|random
    tol: float = 1e-7,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (modes[K,T], center_freqs[K])."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]

    # mirror to reduce edge effects
    x_m = np.concatenate([x[T // 2 : 0 : -1], x, x[-2 : T // 2 - 2 : -1]])
    Tm = x_m.shape[0]

    freqs = np.fft.fftfreq(Tm, d=1.0)
    f_hat = np.fft.fft(x_m)

    # analytic signal: keep positive freqs
    H = np.zeros(Tm)
    H[0] = 1
    H[1 : (Tm + 1) // 2] = 2
    f_hat = f_hat * H

    u_hat = np.zeros((K, Tm), dtype=complex)
    omega = np.zeros(K)
    if init == "random":
        omega = np.sort(np.random.rand(K) * 0.5)
    else:
        omega = 0.5 / K * np.arange(K)

    if DC:
        omega[0] = 0.0

    lam = np.zeros(Tm, dtype=complex)

    u_hat_prev = u_hat.copy()

    for _ in range(int(max_iter)):
        u_hat_sum = np.sum(u_hat, axis=0)
        for k in range(K):
            u_hat_sum_minus = u_hat_sum - u_hat[k]
            denom = 1.0 + alpha * (freqs - omega[k]) ** 2
            u_hat[k] = (f_hat - u_hat_sum_minus - lam / 2.0) / denom

            if not (DC and k == 0):
                # update omega
                num = np.sum(freqs * (np.abs(u_hat[k]) ** 2))
                den = np.sum(np.abs(u_hat[k]) ** 2) + 1e-12
                omega[k] = num / den

        u_hat_sum = np.sum(u_hat, axis=0)
        lam = lam + tau * (u_hat_sum - f_hat)

        # convergence
        diff = np.linalg.norm(u_hat - u_hat_prev) / (np.linalg.norm(u_hat_prev) + 1e-12)
        if diff < tol:
            break
        u_hat_prev = u_hat.copy()

    # back to time domain
    u = np.real(np.fft.ifft(u_hat, axis=1))

    # crop to original length
    start = T // 2
    u = u[:, start : start + T]
    return u.astype(np.float32), omega.astype(np.float32)


@SolverRegistry.register(
    name="vmd",
    family="time_frequency",
    description="Variational Mode Decomposition (VMD) for 1D signals.",
    tags=["time_series", "decomposition", "time_frequency"],
)
class VMDSolver(SolverBase):
    def __init__(
        self,
        K: int = 4,
        alpha: float = 2000.0,
        tau: float = 0.0,
        DC: bool = False,
        init: str = "uniform",
        tol: float = 1e-7,
        max_iter: int = 500,
    ):
        super().__init__()
        self.K = int(K)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.DC = bool(DC)
        self.init = str(init)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def forward(self, x: torch.Tensor) -> SolverOutput:
        xt = x.detach().cpu().numpy()
        if xt.ndim == 1:
            xt = xt[None, :]
        B, T = xt.shape
        modes = []
        omegas = []
        for b in range(B):
            m, w = vmd_1d(
                xt[b],
                K=self.K,
                alpha=self.alpha,
                tau=self.tau,
                DC=self.DC,
                init=self.init,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            modes.append(m)
            omegas.append(w)
        result = torch.from_numpy(np.stack(modes, axis=0)).to(x.device)  # (B,K,T)
        return SolverOutput(result=result, losses={}, extras={"omega": np.stack(omegas, axis=0)})
