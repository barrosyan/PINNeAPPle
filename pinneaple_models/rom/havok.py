from __future__ import annotations
"""HAVOK Hankel alternative view of Koopman for chaotic dynamics."""
from typing import Dict, Optional

import torch

from .base import ROMBase, ROMOutput
from .dmd import DynamicModeDecomposition


class HAVOK(ROMBase):
    """
    HAVOK (MVP):
      - build Hankel (delay embedding) from scalar or low-dim observations
      - apply DMD in embedded space
      - reconstruct embedded trajectory (and optionally map back)

    Notes:
      Full HAVOK also separates forcing term; this MVP focuses on the
      delay-embedding + linear model core.
    """
    def __init__(self, delays: int = 50, r: int = 64, center: bool = True):
        super().__init__()
        self.delays = int(delays)
        self.dmd = DynamicModeDecomposition(r=r, center=center)

    def _hankel(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,D) -> H: (T-delays+1, delays*D)
        T, D = x.shape
        L = self.delays
        if T < L:
            raise ValueError(f"Need T>=delays. Got T={T}, delays={L}")
        rows = []
        for t in range(T - L + 1):
            rows.append(x[t:t+L, :].reshape(-1))
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "HAVOK":
        # X: (T,D) or (B,T,D)
        if X.ndim == 2:
            Xb = X[None, :, :]
        else:
            Xb = X
        Hs = []
        for b in range(Xb.shape[0]):
            Hs.append(self._hankel(Xb[b]))
        H = torch.cat(Hs, dim=0)  # (N, delays*D)
        # treat H as sequence by chunking is complex; MVP: fit DMD on H as one long sequence
        # simplest: interpret H rows as time steps
        self.dmd.fit(H)
        return self

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        if X.ndim != 2:
            raise ValueError("HAVOK MVP expects X with shape (T,D) for now.")
        H = self._hankel(X)  # (T', F)
        yhat = self.dmd.rollout(H[0:1, :], steps=H.shape[0]-1)[0]  # (T',F)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse"] = self.mse(yhat, H)
            losses["total"] = losses["mse"]

        return ROMOutput(y=yhat, losses=losses, extras={"embedded_dim": H.shape[1]})
