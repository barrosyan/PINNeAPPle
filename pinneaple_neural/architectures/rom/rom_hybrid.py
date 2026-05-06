from __future__ import annotations
"""Hybrid ROM combining projection and neural correction."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput
from .pod import POD
from .opinf import OperatorInference


class ROMHybrid(ROMBase):
    """
    ROM Hybrid (MVP):
      - POD encoder/decoder for fields
      - OperatorInference (latent dynamics)
    """
    def __init__(self, field_dim: int, r: int = 64, l2: float = 1e-6):
        super().__init__()
        self.field_dim = int(field_dim)
        self.pod = POD(r=r, center=True)
        self.dyn = OperatorInference(r=r, l2=l2, use_quadratic=True, use_bias=True)

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "ROMHybrid":
        # X: (B,T,D)
        self.pod.fit(X)
        a = self.pod.encode(X).reshape(X.shape[0], X.shape[1], -1)
        self.dyn.fit(a)
        return self

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        # reconstruct + rollout
        B, T, D = X.shape
        a = self.pod.encode(X).reshape(B, T, -1)
        a_hat = self.dyn(a).y  # (B,T,r)
        X_hat = self.pod.decode(a_hat.reshape(B*T, -1), shape=(B, T, D))

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse"] = self.mse(X_hat, X)
            losses["total"] = losses["mse"]

        return ROMOutput(y=X_hat, losses=losses, extras={"a": a, "a_hat": a_hat})
