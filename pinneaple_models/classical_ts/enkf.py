from __future__ import annotations
from typing import Callable, Dict, Optional, Any

import torch

from .base import ClassicalTSBase, ClassicalTSOutput


class EnsembleKalmanFilter(ClassicalTSBase):
    """
    Ensemble Kalman Filter (EnKF) MVP:

      - Maintain ensemble of states X^k
      - Predict each ensemble through f
      - Update using sample covariances

    User provides:
      f(x,u)->x  (vectorized over ensemble)
      h(x)->y
    """
    def __init__(
        self,
        f: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        h: Callable[[torch.Tensor], torch.Tensor],
        Q: torch.Tensor,
        R: torch.Tensor,
        *,
        ensemble_size: int = 64,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

        self.ensemble_size = int(ensemble_size)
        if self.ensemble_size < 2:
            raise ValueError("EnKF requires ensemble_size >= 2")

    def forward(
        self,
        y: torch.Tensor,                 # (B,T,m)
        *,
        u: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,  # (B,n)
        return_ensembles: bool = False,
    ) -> ClassicalTSOutput:
        Bsz, T, m = y.shape
        device, dtype = y.device, y.dtype

        Q = self.Q.to(device=device, dtype=dtype)
        R = self.R.to(device=device, dtype=dtype)

        n = Q.shape[0]
        M = self.ensemble_size

        if x0 is None:
            x_mean = torch.zeros((Bsz, n), device=device, dtype=dtype)
        else:
            x_mean = x0.to(device=device, dtype=dtype)

        # init ensemble: mean + noise
        Lq = torch.linalg.cholesky(Q + 1e-9 * torch.eye(n, device=device, dtype=dtype))
        X = x_mean[:, None, :] + torch.randn((Bsz, M, n), device=device, dtype=dtype) @ Lq.transpose(-1, -2)

        xs = []
        ens_store = []

        for t in range(T):
            ut = None if u is None else u[:, t, :]

            # predict ensemble
            Xflat = X.reshape(Bsz * M, n)
            if ut is None:
                Xpred = self.f(Xflat, None).view(Bsz, M, n)
            else:
                Urep = ut[:, None, :].expand(Bsz, M, ut.shape[-1]).reshape(Bsz * M, -1)
                Xpred = self.f(Xflat, Urep).view(Bsz, M, n)

            # add process noise
            Xpred = Xpred + torch.randn_like(Xpred) @ Lq.transpose(-1, -2)

            # predicted obs
            Ypred = self.h(Xpred.reshape(Bsz * M, n)).view(Bsz, M, m)

            xbar = Xpred.mean(dim=1)  # (B,n)
            ybar = Ypred.mean(dim=1)  # (B,m)

            dX = Xpred - xbar[:, None, :]
            dY = Ypred - ybar[:, None, :]

            # covariances (unbiased sample cov)
            Pxy = (dX.transpose(1, 2) @ dY) / (M - 1)  # (B,n,m)
            Pyy = (dY.transpose(1, 2) @ dY) / (M - 1)  # (B,m,m)
            Pyy = Pyy + R

            # K = Pxy @ inv(Pyy)  (but use solve for numerical stability)
            # Solve: Pyy^T K^T = Pxy^T  => K = (solve(Pyy, Pxy^T))^T
            K = torch.linalg.solve(Pyy, Pxy.transpose(1, 2)).transpose(1, 2)  # (B,n,m)

            # update ensemble with perturbed obs (stochastic EnKF)
            Lr = torch.linalg.cholesky(R + 1e-9 * torch.eye(m, device=device, dtype=dtype))
            eps = torch.randn((Bsz, M, m), device=device, dtype=dtype) @ Lr.transpose(-1, -2)
            y_pert = y[:, t, :][:, None, :].expand(Bsz, M, m) + eps

            X = Xpred + torch.einsum("bnm,bkm->bkn", K, (y_pert - Ypred))

            xs.append(X.mean(dim=1))
            if return_ensembles:
                ens_store.append(X)

        extras: Dict[str, Any] = {}
        if return_ensembles:
            extras["ensembles"] = torch.stack(ens_store, dim=1)  # (B,T,M,n)

        return ClassicalTSOutput(
            y=torch.stack(xs, dim=1),
            losses={"total": torch.tensor(0.0, device=device)},
            extras=extras,
        )
