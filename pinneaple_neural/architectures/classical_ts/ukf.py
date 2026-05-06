from __future__ import annotations
"""Unscented Kalman filter for nonlinear state estimation."""
from typing import Callable, Dict, Optional, Any

import torch

from .base import ClassicalTSBase, ClassicalTSOutput


class UnscentedKalmanFilter(ClassicalTSBase):
    """
    UKF (sigma-point filter), batch-friendly.

    User provides:
      f(x,u)->x
      h(x)->y

    Parameters:
      alpha, beta, kappa control sigma points.

    Notes (robustness vs. naïve UKF):
      - avoids explicit matrix inverse (uses solve)
      - enforces symmetry and adds jitter to keep covariances SPD
      - stable Cholesky with retry
    """
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        Q: torch.Tensor,
        R: torch.Tensor,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        jitter: float = 1e-6,
        jitter_growth: float = 10.0,
        max_cholesky_tries: int = 5,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.jitter = float(jitter)
        self.jitter_growth = float(jitter_growth)
        self.max_cholesky_tries = int(max_cholesky_tries)

    @staticmethod
    def _symmetrize(A: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A + A.transpose(-1, -2))

    def _stable_cholesky(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (B,n,n) expected symmetric PSD/SPD.
        Returns L such that L L^T = A (+ jitter I if needed)
        """
        B, n, _ = A.shape
        I = torch.eye(n, device=A.device, dtype=A.dtype).expand(B, n, n)
        Aj = self._symmetrize(A)

        eps = self.jitter
        last_err: Optional[Exception] = None
        for _ in range(self.max_cholesky_tries):
            try:
                return torch.linalg.cholesky(Aj + eps * I)
            except Exception as e:
                last_err = e
                eps *= self.jitter_growth

        # if it still fails, raise a helpful error
        raise RuntimeError(
            f"Cholesky failed after {self.max_cholesky_tries} tries. "
            f"Last error: {last_err}"
        )

    def _sigma_points(self, x: torch.Tensor, P: torch.Tensor):
        # x: (B,n), P:(B,n,n)
        B, n = x.shape
        device, dtype = x.device, x.dtype

        lam = (self.alpha ** 2) * (n + self.kappa) - n
        c = n + lam
        if c <= 0:
            raise ValueError(
                f"Invalid sigma-point scaling: c=n+lambda must be > 0, got c={c}. "
                "Try increasing alpha or kappa."
            )

        # Stable Cholesky of (c * P)
        S = self._stable_cholesky((c * P).to(dtype=dtype))  # (B,n,n), lower-tri

        pts = [x]
        for i in range(n):
            col = S[:, :, i]
            pts.append(x + col)
            pts.append(x - col)
        X = torch.stack(pts, dim=1)  # (B, 2n+1, n)

        Wm = torch.full((2 * n + 1,), 1.0 / (2.0 * c), device=device, dtype=dtype)
        Wc = Wm.clone()
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)

        return X, Wm, Wc

    def forward(
        self,
        y: torch.Tensor,                  # (B,T,m)
        *,
        u: Optional[torch.Tensor] = None,  # (B,T,du)
        x0: Optional[torch.Tensor] = None, # (B,n) or (n,)
        P0: Optional[torch.Tensor] = None, # (B,n,n) or (n,n)
        return_gain: bool = False,
    ) -> ClassicalTSOutput:
        Bsz, T, m = y.shape
        device, dtype = y.device, y.dtype

        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)

        # basic shape validation
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be (n,n). Got shape {tuple(Q.shape)}.")
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be (m,m). Got shape {tuple(R.shape)}.")
        if R.shape[0] != m:
            raise ValueError(f"R shape {tuple(R.shape)} incompatible with y last dim m={m}.")

        n = Q.shape[0]

        if x0 is None:
            x = torch.zeros((Bsz, n), device=device, dtype=dtype)
        else:
            x = x0.to(device=device, dtype=dtype)
            if x.ndim == 1:
                if x.shape[0] != n:
                    raise ValueError(f"x0 has dim {x.shape[0]} but expected n={n}.")
                x = x.unsqueeze(0).expand(Bsz, n).clone()
            elif x.ndim == 2:
                if x.shape[1] != n:
                    raise ValueError(f"x0 second dim {x.shape[1]} but expected n={n}.")
                if x.shape[0] != Bsz:
                    x = x.expand(Bsz, n).clone()
            else:
                raise ValueError(f"x0 must be (n,) or (B,n). Got shape {tuple(x.shape)}.")

        if P0 is None:
            P = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n).clone()
        else:
            P = P0.to(device=device, dtype=dtype)
            if P.ndim == 2:
                if P.shape != (n, n):
                    raise ValueError(f"P0 must be (n,n). Got shape {tuple(P.shape)}.")
                P = P.unsqueeze(0).expand(Bsz, n, n).clone()
            elif P.ndim == 3:
                if P.shape[1:] != (n, n):
                    raise ValueError(f"P0 must be (B,n,n). Got shape {tuple(P.shape)}.")
                if P.shape[0] != Bsz:
                    P = P.expand(Bsz, n, n).clone()
            else:
                raise ValueError(f"P0 must be (n,n) or (B,n,n). Got shape {tuple(P.shape)}.")

        # ensure symmetric initial covariance
        P = self._symmetrize(P)

        xs, Ps, Ks = [], [], []

        # precompute identity for jitter/sym (per batch)
        I = torch.eye(n, device=device, dtype=dtype).expand(Bsz, n, n)

        for t in range(T):
            ut = None if u is None else u[:, t, :]

            # sigma points
            X, Wm, Wc = self._sigma_points(x, P)  # X: (B,S,n)
            Spts = X.shape[1]

            # predict through dynamics
            Xflat = X.reshape(-1, n)
            if ut is None:
                Xp_flat = self.f(Xflat, None)
            else:
                # (B,du) -> (B*S,du) matching flattened sigma points
                urep = ut.repeat_interleave(Spts, dim=0)
                Xp_flat = self.f(Xflat, urep)

            Xp = Xp_flat.view(Bsz, Spts, n)  # (B,S,n)

            x_pred = torch.sum(Xp * Wm[None, :, None], dim=1)  # (B,n)
            dX = Xp - x_pred[:, None, :]                     # (B,S,n)

            # P_pred = sum_i Wc[i] dX_i dX_i^T + Q
            P_pred = torch.sum(
                Wc[None, :, None, None] * (dX[:, :, :, None] @ dX[:, :, None, :]),
                dim=1
            ) + Q

            P_pred = self._symmetrize(P_pred) + self.jitter * I

            # predict observation
            Yp = self.h(Xp.reshape(-1, n)).view(Bsz, Spts, m)  # (B,S,m)
            y_pred = torch.sum(Yp * Wm[None, :, None], dim=1)  # (B,m)
            dY = Yp - y_pred[:, None, :]                       # (B,S,m)

            # innovation covariance S = sum_i Wc[i] dY_i dY_i^T + R
            Syy = torch.sum(
                Wc[None, :, None, None] * (dY[:, :, :, None] @ dY[:, :, None, :]),
                dim=1
            ) + R
            Syy = self._symmetrize(Syy)

            # stabilize Syy too (helps solve)
            Im = torch.eye(m, device=device, dtype=dtype).expand(Bsz, m, m)
            Syy = Syy + self.jitter * Im

            # cross-cov Pxy = sum_i Wc[i] dX_i dY_i^T
            Pxy = torch.sum(
                Wc[None, :, None, None] * (dX[:, :, :, None] @ dY[:, :, None, :]),
                dim=1
            )  # (B,n,m)

            # Kalman gain: K = Pxy S^{-1}  (avoid inv -> solve)
            # Solve: Syy^T * K^T = Pxy^T
            K = torch.linalg.solve(Syy.transpose(-1, -2), Pxy.transpose(-1, -2)).transpose(-1, -2)  # (B,n,m)

            innov = y[:, t, :] - y_pred  # (B,m)
            x = x_pred + (K @ innov.unsqueeze(-1)).squeeze(-1)  # (B,n)

            # Cov update (Joseph form is optional; here: standard + symmetrize/jitter)
            P = P_pred - K @ Syy @ K.transpose(-1, -2)
            P = self._symmetrize(P) + self.jitter * I

            xs.append(x)
            Ps.append(P)
            if return_gain:
                Ks.append(K)

        extras: Dict[str, Any] = {
            "P": torch.stack(Ps, dim=1),
        }
        if return_gain:
            extras["K"] = torch.stack(Ks, dim=1)

        return ClassicalTSOutput(
            y=torch.stack(xs, dim=1),
            losses={"total": torch.tensor(0.0, device=device, dtype=dtype)},
            extras=extras,
        )
