from __future__ import annotations
"""Hybrid RBF network combining linear and nonlinear features."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RCBase, RCOutput
from .rbf import RBFNetwork


class HybridRBFNetwork(RCBase):
    """
    Hybrid RBF Network:
      y = [Phi(x), x, 1] W
    where W is solved by ridge regression in a single linear system.

    Supports "selective regularization" (e.g., do not regularize bias/linear)
    WITHOUT changing RCBase.ridge_solve, via feature column scaling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_centers: int = 512,
        sigma: Optional[float] = None,
        l2: float = 1e-6,
        learn_centers: bool = False,
        use_linear: bool = True,
        use_bias: bool = True,
        # New knobs (do not touch ridge_solve):
        reg_linear: bool = True,
        reg_bias: bool = False,
        # safety/behavior:
        require_fitted: bool = True,
        no_reg_scale: float = 1e6,
        feature_mode: Literal["rbf_first"] = "rbf_first",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.l2 = float(l2)

        self.use_linear = bool(use_linear)
        self.use_bias = bool(use_bias)

        self.reg_linear = bool(reg_linear)
        self.reg_bias = bool(reg_bias)

        self.require_fitted = bool(require_fitted)
        self.no_reg_scale = float(no_reg_scale)

        self.rbf = RBFNetwork(
            in_dim=in_dim,
            out_dim=out_dim,
            num_centers=num_centers,
            sigma=sigma,
            l2=l2,
            learn_centers=learn_centers,
        )

        self.num_centers = int(num_centers)

        feat_dim = self.num_centers + (self.in_dim if self.use_linear else 0) + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(feat_dim, out_dim), requires_grad=False)

        # will store the per-column scaling used to emulate selective ridge
        self.register_buffer("_col_scale", torch.ones(feat_dim), persistent=False)

        self._fitted = False

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        Phi = self.rbf._phi(x)  # (N, M)
        feats = [Phi]
        if self.use_linear:
            feats.append(x)
        if self.use_bias:
            feats.append(torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
        return torch.cat(feats, dim=-1)

    def _build_col_scale(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build diagonal scaling vector d (length F) such that running:
            W' = ridge_solve(X * d, Y, l2=self.l2)
            W  = d * W'
        emulates different regularization strengths per column.

        If we set d_j very large, effective penalty l2 / d_j^2 ~ 0 (almost unregularized).
        """
        feat_dim = self.W_out.shape[0]
        d = torch.ones((feat_dim,), device=device, dtype=dtype)

        # Indices layout: [RBF (0..M-1), LINEAR (next in_dim), BIAS (last)]
        idx = 0

        # RBF block always uses d=1 (regularized normally)
        idx += self.num_centers

        # Linear block
        if self.use_linear:
            if not self.reg_linear:
                d[idx : idx + self.in_dim] = self.no_reg_scale
            idx += self.in_dim

        # Bias
        if self.use_bias:
            if not self.reg_bias:
                d[idx] = self.no_reg_scale
            idx += 1

        return d

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, *, init_centers_from_data: bool = True) -> "HybridRBFNetwork":
        # Init centers + sigma via underlying RBF.
        # NOTE: this may also solve an internal output inside RBFNetwork; harmless but may be wasted work.
        self.rbf.fit(x, y, init_centers_from_data=init_centers_from_data)

        F = self._features(x)  # (N, feat_dim)

        # Build scaling to emulate selective regularization
        d = self._build_col_scale(device=F.device, dtype=F.dtype)  # (feat_dim,)
        self._col_scale = d

        # Apply scaling to columns: X' = X * d
        # (broadcast d across rows)
        F_scaled = F * d.unsqueeze(0)

        # Solve standard ridge on scaled design
        W_scaled = self.ridge_solve(F_scaled, y, l2=self.l2)  # (feat_dim, out_dim)

        # Map back: W = d * W_scaled
        W = d.unsqueeze(1) * W_scaled

        self.W_out.copy_(W)
        self._fitted = True
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RCOutput:
        if self.require_fitted and not self._fitted:
            raise RuntimeError("HybridRBFNetwork.forward() called before fit().")

        F = self._features(x)
        y = F @ self.W_out

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y, losses=losses, extras={"fitted": self._fitted})