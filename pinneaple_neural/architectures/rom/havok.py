from __future__ import annotations
"""HAVOK Hankel alternative view of Koopman for chaotic dynamics."""
from typing import Dict, Optional, Tuple

import torch

from .base import ROMBase, ROMOutput
from .dmd import DynamicModeDecomposition


class HAVOK(ROMBase):
    """
    HAVOK (MVP-ish, closer to literature):
      1) build Hankel (delay embedding) from scalar or low-dim observations
      2) center + SVD of Hankel and project to low-rank subspace (HAVOK core)
      3) fit linear dynamics in that subspace via DMD (no forced stitching across trajectories)
      4) rollout in subspace; optionally decode to observation space (simple heuristic)

    Notes:
      - Full HAVOK often separates forcing term; this version keeps the linear core.
      - For multi-trajectory fit, we accumulate (X0, X1) pairs without introducing fake transitions.
    """

    def __init__(
        self,
        delays: int = 50,
        r: int = 64,
        center: bool = True,
        svd_eps: float = 1e-12,
        decode_mode: str = "last",  # "last" or "none"
    ):
        super().__init__()
        self.delays = int(delays)
        self.r = int(r)
        self.center = bool(center)
        self.svd_eps = float(svd_eps)
        self.decode_mode = str(decode_mode)

        # DMD will be fit on the reduced coordinates (Hr), so r here is usually <= self.r anyway.
        self.dmd = DynamicModeDecomposition(r=self.r, center=False)

        # learned during fit
        self._mu: Optional[torch.Tensor] = None        # (F,)
        self._Vh_r: Optional[torch.Tensor] = None      # (r, F)  where F = delays*D
        self._D: Optional[int] = None                  # observation dim

    def _hankel(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,D) -> H: (T-delays+1, delays*D)
        T, D = x.shape
        L = self.delays
        if T < L:
            raise ValueError(f"Need T>=delays. Got T={T}, delays={L}")
        rows = []
        for t in range(T - L + 1):
            rows.append(x[t : t + L, :].reshape(-1))
        return torch.stack(rows, dim=0)

    def _fit_hankel_svd(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fit SVD basis for Hankel and return (Hc, mu, Vh_r).
        H: (N,F)
        """
        if self.center:
            mu = H.mean(dim=0, keepdim=False)
            Hc = H - mu
        else:
            mu = torch.zeros(H.shape[1], device=H.device, dtype=H.dtype)
            Hc = H

        # SVD: Hc = U S Vh
        # We keep Vh_r to project new Hankel rows into reduced coordinates.
        U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)

        # pick effective rank: min(self.r, rank(Hc))
        # Guard: singular values can be tiny; we still cap by self.r
        r_eff = min(self.r, Vh.shape[0])
        # optionally shrink further if numerical rank is smaller
        if self.svd_eps > 0:
            # keep svs above eps * max_sv
            max_sv = S[0].clamp_min(self.svd_eps)
            keep = (S / max_sv) > self.svd_eps
            r_num = int(keep.sum().item())
            r_eff = max(1, min(r_eff, r_num))

        Vh_r = Vh[:r_eff, :]  # (r_eff, F)
        return Hc, mu, Vh_r

    def _project(self, H: torch.Tensor) -> torch.Tensor:
        """
        Project Hankel rows H (N,F) to reduced coords Z (N,r_eff).
        Uses learned (mu, Vh_r).
        """
        if self._mu is None or self._Vh_r is None:
            raise RuntimeError("HAVOK not fitted: missing SVD basis.")
        Hc = H - self._mu if self.center else H
        # Z = Hc @ V_r, where V_r = Vh_r^T
        return Hc @ self._Vh_r.T

    def _lift(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Lift reduced coords Z (N,r_eff) back to Hankel space Hhat (N,F) via truncated SVD basis.
        Hhat = Z @ Vh_r + mu
        """
        if self._mu is None or self._Vh_r is None:
            raise RuntimeError("HAVOK not fitted: missing SVD basis.")
        Hc_hat = Z @ self._Vh_r
        return Hc_hat + (self._mu if self.center else 0.0)

    def _decode_observation(self, Hhat: torch.Tensor) -> torch.Tensor:
        """
        Simple heuristic decode from Hankel-row back to observation:
        take the *last* D entries (corresponding to the last delay step).
        Hhat: (N, delays*D) -> Xhat: (N, D)
        """
        if self._D is None:
            raise RuntimeError("HAVOK not fitted: missing observation dim.")
        if self.decode_mode == "none":
            raise RuntimeError("decode_mode='none' (no decoding configured).")
        D = self._D
        return Hhat[:, -D:]

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "HAVOK":
        """
        X: (T,D) or (B,T,D)
        """
        if X.ndim == 2:
            Xb = X[None, :, :]
        elif X.ndim == 3:
            Xb = X
        else:
            raise ValueError("X must have shape (T,D) or (B,T,D).")

        self._D = int(Xb.shape[-1])

        # 1) Build Hankel rows for all trajectories (no temporal stitching in the DMD fit)
        Hb_list = [self._hankel(Xb[b]) for b in range(Xb.shape[0])]
        H_all = torch.cat(Hb_list, dim=0)  # used only to fit SVD basis

        # 2) Fit Hankel SVD basis (HAVOK core)
        _, mu, Vh_r = self._fit_hankel_svd(H_all)
        self._mu = mu
        self._Vh_r = Vh_r

        # 3) Build reduced sequences and accumulate pairs (Z0, Z1) without fake transitions
        Z0_chunks = []
        Z1_chunks = []
        for Hb in Hb_list:
            Zb = self._project(Hb)  # (Tb', r_eff)
            if Zb.shape[0] < 2:
                continue
            Z0_chunks.append(Zb[:-1, :])
            Z1_chunks.append(Zb[1:, :])

        if not Z0_chunks:
            raise ValueError("Not enough time steps after Hankel embedding to fit dynamics.")

        Z0 = torch.cat(Z0_chunks, dim=0)
        Z1 = torch.cat(Z1_chunks, dim=0)

        # 4) Fit DMD using paired snapshots (recommended for multi-trajectory)
        # Prefer a dedicated method if your DMD has it; otherwise, fall back to fitting on a pseudo-sequence.
        if hasattr(self.dmd, "fit_pairs"):
            self.dmd.fit_pairs(Z0, Z1)
        else:
            # fallback: approximate by fitting on a stacked sequence (still no stitching *within* pairs)
            Z_seq = torch.cat([Z0, Z1[-1:, :]], dim=0)
            self.dmd.fit(Z_seq)

        return self

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        if X.ndim != 2:
            raise ValueError("HAVOK expects X with shape (T,D).")

        # Hankel embedding
        H = self._hankel(X)  # (T', F)
        Z = self._project(H)  # (T', r_eff)

        # rollout in reduced space
        # We expect rollout returns (batch, time, r_eff) or similar; handle common cases robustly.
        Z0 = Z[0:1, :]  # (1, r_eff)
        Zhat = self.dmd.rollout(Z0, steps=Z.shape[0] - 1)

        # normalize output shape to (T', r_eff)
        if isinstance(Zhat, (tuple, list)):
            Zhat = Zhat[0]
        # common patterns: (1,T',r) or (T',r)
        if Zhat.ndim == 3:
            Zhat = Zhat[0]
        if Zhat.shape[0] != Z.shape[0]:
            # if rollout returned only future steps (T'-1), prepend initial
            if Zhat.shape[0] == Z.shape[0] - 1:
                Zhat = torch.cat([Z0, Zhat], dim=0)
            else:
                raise RuntimeError(f"Unexpected rollout shape {tuple(Zhat.shape)}; expected T'={Z.shape[0]}")

        # lift back to Hankel space (optional, but useful for loss/debug)
        Hhat = self._lift(Zhat)  # (T', F)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse_embedded"] = self.mse(Zhat, Z)
            losses["mse_hankel"] = self.mse(Hhat, H)
            # choose what you consider "total" (usually embedded is enough)
            losses["total"] = losses["mse_embedded"]

        extras: Dict[str, torch.Tensor | int | str] = {
            "embedded_dim": int(Z.shape[1]),
            "hankel_dim": int(H.shape[1]),
            "delays": int(self.delays),
        }

        # Main output:
        # - By default, keep y as reduced coords (closer to ROM output)
        # - Also provide decoded observation (heuristic) if enabled
        y_out = Zhat
        if self.decode_mode != "none":
            Xhat = self._decode_observation(Hhat)  # (T',D)
            extras["xhat"] = Xhat
            extras["space"] = "reduced+decoded"
        else:
            extras["space"] = "reduced"

        return ROMOutput(y=y_out, losses=losses, extras=extras)