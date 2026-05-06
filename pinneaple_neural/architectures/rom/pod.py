from __future__ import annotations
"""Proper orthogonal decomposition via SVD for ROM."""

from typing import Dict, Optional, Tuple

import torch

from .base import ROMBase, ROMOutput


class POD(ROMBase):
    """
    POD via SVD (equivalente a PCA linear em espaço de estados) para ROM.

    Fit:
      snapshots X: (N_snap, D) or (B,T,D) -> flattened to (N, D)

    Produz:
      mean_:  (1, D) média (se center=True)
      basis_: (D, r_eff) base POD (modos) com r_eff <= min(N, D)
      a:      (N, r_eff) coeficientes

    Reconstrói:
      X_hat = a basis_^T + mean_
    """

    def __init__(
        self,
        r: int = 64,
        center: bool = True,
        energy: Optional[float] = None,  # ex: 0.999 para escolher r por energia
    ):
        """
        Parameters
        ----------
        r : int
            Número máximo de modos (usado se energy is None).
        center : bool
            Se True, centraliza os snapshots subtraindo a média.
        energy : Optional[float]
            Se definido (0 < energy <= 1), escolhe r automaticamente tal que
            sum_{i<=r} S_i^2 / sum S_i^2 >= energy. Ignora `r` como valor fixo.
        """
        super().__init__()
        self.r = int(r)
        self.center = bool(center)
        self.energy = None if energy is None else float(energy)

        if self.r <= 0:
            raise ValueError("r must be a positive integer.")
        if self.energy is not None and not (0.0 < self.energy <= 1.0):
            raise ValueError("energy must be in (0, 1].")

        # Buffers vazios (evita shapes errados antes do fit)
        self.register_buffer("mean_", torch.empty(0))   # (1, D) após fit
        self.register_buffer("basis_", torch.empty(0))  # (D, r_eff) após fit
        self._fitted = False
        self._r_eff: int = 0

    @staticmethod
    def _flatten_snapshots(X: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int, int]]]:
        """(B,T,D)->(B*T,D) e retorna shape original para possível reshape."""
        if X.ndim == 3:
            B, T, D = X.shape
            return X.reshape(B * T, D), (B, T, D)
        if X.ndim == 2:
            return X, None
        raise ValueError(f"Expected X with ndim 2 or 3, got {X.ndim}.")

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "POD":
        X2, _ = self._flatten_snapshots(X)  # (N,D)

        if self.center:
            mu = X2.mean(dim=0, keepdim=True)  # (1,D)
            Xc = X2 - mu
            self.mean_ = mu
        else:
            self.mean_ = torch.zeros((1, X2.shape[1]), device=X2.device, dtype=X2.dtype)
            Xc = X2

        # SVD: Xc = U S V^T -> bases POD = colunas de V
        # torch.linalg.svd retorna Vh = V^T
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

        # Define r_eff (fixo por r ou por energia)
        max_r = Vh.shape[0]  # = min(N, D)
        if self.energy is not None:
            # energia proporcional a S^2
            s2 = S**2
            denom = s2.sum().clamp_min(torch.finfo(s2.dtype).eps)
            cum = torch.cumsum(s2, dim=0) / denom
            # menor r tal que cum[r-1] >= energy
            r_eff = int(torch.searchsorted(cum, torch.tensor(self.energy, device=cum.device, dtype=cum.dtype)).item()) + 1
            r_eff = max(1, min(r_eff, max_r))
        else:
            r_eff = min(self.r, max_r)

        Vr = Vh[:r_eff, :].t().contiguous()  # (D, r_eff)

        self.basis_ = Vr
        self._r_eff = r_eff
        self._fitted = True
        return self

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted. Call fit(X) first.")
        X2, _ = self._flatten_snapshots(X)  # (N,D)
        Xc = X2 - self.mean_
        a = Xc @ self.basis_  # (N,r_eff)
        return a

    def decode(self, a: torch.Tensor, *, shape: Optional[tuple] = None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted. Call fit(X) first.")
        Xc = a @ self.basis_.t()  # (N,D)
        X = Xc + self.mean_
        if shape is not None:
            X = X.reshape(*shape)
        return X

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        X2, orig_shape = self._flatten_snapshots(X)

        a = self.encode(X2)  # encode aceita 2D também
        Xhat2 = self.decode(a)  # (N,D)

        # volta para (B,T,D) se necessário
        Xhat = Xhat2.reshape(*orig_shape) if orig_shape is not None else Xhat2

        losses: Dict[str, torch.Tensor] = {}
        if return_loss:
            mse = self.mse(Xhat, X)
            losses = {"mse": mse, "total": mse}

        return ROMOutput(
            y=Xhat,
            losses=losses,
            extras={
                "a": a,
                "basis": self.basis_,
                "mean": self.mean_,
                "fitted": self._fitted,
                "r_eff": self._r_eff,
                "center": self.center,
                "energy": self.energy,
            },
        )