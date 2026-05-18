from __future__ import annotations
"""TFT-lite (strong): GRN + grouped variable selection + causal encoder + cross-attentive decoder."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeSeriesModelBase, TSOutput


# ============================================================
# Utils
# ============================================================

def _causal_mask(L: int, device: torch.device) -> torch.Tensor:
    # True = masked in PyTorch Transformer API
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


# ============================================================
# GRN (lite, but useful)
# ============================================================

class GRN(nn.Module):
    """
    Gated Residual Network (lite):
      y = LayerNorm(x + sigmoid(Wg x) * Drop( W2 gelu(W1 x) ))
    """
    def __init__(self, d: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class GLU(nn.Module):
    """Gated Linear Unit used for skip gating."""
    def __init__(self, d: int):
        super().__init__()
        self.fc = nn.Linear(d, 2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


# ============================================================
# Grouped Variable Selection (VSN-lite)
# ============================================================

class GroupVSN(nn.Module):
    """
    Variable selection network (lite) over groups, not per-variable.

    Given group tensors g_i each shaped (B,L,d_model), produce weights w_i(t)
    and output z(t)=Σ w_i(t) * g_i(t).

    This matches TFT spirit (dynamic selection) without per-feature embeddings.
    """
    def __init__(self, d_model: int, num_groups: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.num_groups = int(num_groups)
        self.score_grn = GRN(d_model, hidden=hidden, dropout=dropout)
        self.score = nn.Linear(d_model, num_groups)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, groups: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        groups: tuple of (B,L,d_model), length = num_groups
        returns:
          z: (B,L,d_model)
          w: (B,L,num_groups)
        """
        if len(groups) != self.num_groups:
            raise ValueError(f"Expected {self.num_groups} groups, got {len(groups)}")

        # context for scoring: sum of groups (cheap, stable)
        ctx = torch.stack(groups, dim=0).sum(dim=0)  # (B,L,d_model)
        ctx = self.score_grn(ctx)
        logits = self.score(ctx)  # (B,L,G)
        w = torch.softmax(logits, dim=-1)
        w = self.drop(w)

        # weighted sum
        z = 0.0
        for i, gi in enumerate(groups):
            z = z + gi * w[..., i:i+1]
        return z, w


# ============================================================
# TFT-lite Strong Model
# ============================================================

class TemporalFusionTransformer(TimeSeriesModelBase):
    """
    TFT-lite (strong, honest):
      - Input projection
      - Grouped variable selection (past groups + optional future group)
      - Causal TransformerEncoder over past
      - TransformerDecoder with learnable future queries (multi-step) attending to past memory
      - GRN/GLU gating around blocks for stability and TFT-like behavior

    Inputs:
      x_past:  (B, L, in_dim)
      x_future:(B, H, in_dim) optional, used as future covariates group if provided
    Output:
      y_hat: (B, H, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)

        # Projections
        self.past_proj = nn.Linear(in_dim, d_model)
        self.future_proj = nn.Linear(in_dim, d_model)

        # Group selection:
        # - group0: all past features projected
        # - group1: a gated transform of past (acts like "target/past-cov" separation proxy)
        # - group2: optional future features projected (if x_future provided)
        self.past_grn = GRN(d_model, hidden=2 * d_model, dropout=dropout)
        self.past_gate = GLU(d_model)

        # VSN-lite: if no future group at runtime, we fall back to 2-group selection.
        self.vsn2 = GroupVSN(d_model, num_groups=2, hidden=2 * d_model, dropout=dropout)
        self.vsn3 = GroupVSN(d_model, num_groups=3, hidden=2 * d_model, dropout=dropout)

        # Encoder (causal)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_enc_layers))
        self.enc_post = GRN(d_model, hidden=2 * d_model, dropout=dropout)

        # Decoder: learnable queries + cross-attention into encoder memory
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_dec_layers))
        self.dec_post = GRN(d_model, hidden=2 * d_model, dropout=dropout)

        # Learnable future queries (H tokens)
        self.future_query = nn.Parameter(torch.zeros(1, self.horizon, d_model))
        nn.init.normal_(self.future_query, mean=0.0, std=0.02)

        # If you have x_future, inject it into decoder input (TFT-ish)
        self.future_inject = GRN(d_model, hidden=2 * d_model, dropout=dropout)

        # Output
        self.out = nn.Linear(d_model, out_dim)

    def forward(
        self,
        x_past: torch.Tensor,
        *,
        x_future: Optional[torch.Tensor] = None,
        y_future: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> TSOutput:
        """
        x_past:   (B, L, in_dim)
        x_future: (B, H, in_dim) optional
        """
        B, L, _ = x_past.shape
        H = self.horizon

        # -------------------------
        # 1) Past groups
        # -------------------------
        p0 = self.past_proj(x_past)          # (B,L,d)
        p1 = self.past_grn(p0)               # (B,L,d)
        p1 = self.past_gate(p1)              # (B,L,d) gated transform

        # VSN-lite over past (+ optional future)
        if x_future is not None:
            if x_future.shape[1] != H:
                raise ValueError(f"x_future length {x_future.shape[1]} must match horizon {H}.")
            f0 = self.future_proj(x_future)  # (B,H,d)
            # For selection we need aligned length; use past-only selection for encoder,
            # and pass future covariates later into decoder input.
            enc_in, w_past = self.vsn2((p0, p1))  # (B,L,d), (B,L,2)
        else:
            enc_in, w_past = self.vsn2((p0, p1))

        # -------------------------
        # 2) Causal Encoder
        # -------------------------
        src_mask = _causal_mask(L, device=x_past.device)  # (L,L) bool
        mem = self.encoder(enc_in, mask=src_mask)         # (B,L,d)
        mem = self.enc_post(mem)

        # -------------------------
        # 3) Decoder input: learnable queries (+ optional future covariates injection)
        # -------------------------
        tgt = self.future_query.expand(B, H, -1)  # (B,H,d)

        if x_future is not None:
            f0 = self.future_proj(x_future)      # (B,H,d)
            tgt = self.future_inject(tgt + f0)   # inject future covariates into query tokens

        # decoder self-attn should be causal too (multi-step generation)
        tgt_mask = _causal_mask(H, device=x_past.device)

        dec = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)  # (B,H,d)
        dec = self.dec_post(dec)

        y_hat = self.out(dec)  # (B,H,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_hat.device)}
        if return_loss and y_future is not None:
            losses["mse"] = self.mse(y_hat, y_future)
            losses["total"] = losses["mse"]

        extras = {
            "memory": mem,
            "vsn_weights_past": w_past,  # (B,L,2)
        }
        return TSOutput(y=y_hat, losses=losses, extras=extras)