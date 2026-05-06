"""Temporal Fusion Transformer (Lim et al., 2021) — simplified, GPU-ready."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import TSModelBase, TSOutput


@dataclass
class TFTConfig:
    input_len:      int   = 64
    horizon:        int   = 16
    n_features:     int   = 1
    n_targets:      int   = 1
    hidden_size:    int   = 64
    num_heads:      int   = 4
    num_lstm_layers: int  = 2
    dropout:        float = 0.1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _GLU(nn.Module):
    """Gated Linear Unit: splits last dim in half, gates with sigmoid."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        x1, x2 = h.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class _GRN(nn.Module):
    """Gated Residual Network."""
    def __init__(self, d: int, d_context: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.fc1   = nn.Linear(d, d)
        self.fc2   = nn.Linear(d, d)
        self.ctx   = nn.Linear(d_context, d, bias=False) if d_context else None
        self.gate  = _GLU(d, d)
        self.norm  = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        if ctx is not None and self.ctx is not None:
            h = h + self.ctx(ctx)
        h = self.drop(self.fc2(h))
        h = self.gate(h)
        return self.norm(h + x)


class _VariableSelectionNetwork(nn.Module):
    """Selects and weights input variables via a GRN + softmax gate."""
    def __init__(self, n_vars: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars   = n_vars
        self.projections = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        self.grn_flat = _GRN(n_vars * d, dropout=dropout)
        self.weight   = nn.Linear(n_vars * d, n_vars)
        self.var_grns = nn.ModuleList([_GRN(d, dropout=dropout) for _ in range(n_vars)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_vars)
        B, T, V = x.shape
        embedded = [self.projections[i](x[..., i:i+1]) for i in range(V)]  # list of (B,T,d)
        flat = torch.cat(embedded, dim=-1)                                   # (B,T,V*d)
        weights = torch.softmax(self.weight(self.grn_flat(flat)), dim=-1)   # (B,T,V)
        processed = torch.stack(
            [self.var_grns[i](embedded[i]) for i in range(V)], dim=-2      # (B,T,V,d)
        )
        return (weights.unsqueeze(-1) * processed).sum(dim=-2)              # (B,T,d)


class _TemporalSelfAttention(nn.Module):
    """Interpretable multi-head attention (shared V projection across heads)."""
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, self.d_head)   # shared V
        self.W_o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        Q = self.W_q(x).view(B, T, H, Dh).transpose(1, 2)   # (B,H,T,Dh)
        K = self.W_k(x).view(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).unsqueeze(1).expand(B, H, T, Dh)     # shared V

        scores = Q @ K.transpose(-2, -1) / (Dh ** 0.5)       # (B,H,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn   = self.drop(torch.softmax(scores, dim=-1))
        out    = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# TFT main model
# ---------------------------------------------------------------------------

class TFTForecaster(TSModelBase):
    """
    Temporal Fusion Transformer.
    Input  : (B, L, F)
    Output : TSOutput with y_hat (B, H, n_targets)

    Architecture (simplified TFT without static covariates):
      1. Variable Selection Network  → encoder LSTM
      2. Variable Selection Network  → decoder LSTM  (future-known placeholder)
      3. Static-enrichment GRN
      4. Temporal self-attention on decoder output
      5. Position-wise GRN feed-forward
      6. Quantile / point output head
    """

    def __init__(self, cfg: Optional[TFTConfig] = None, **kw):
        super().__init__()
        self.cfg = cfg or TFTConfig(**kw)
        c = self.cfg
        d = c.hidden_size

        # Encoder path
        self.enc_vsn  = _VariableSelectionNetwork(c.n_features, d, c.dropout)
        self.enc_lstm = nn.LSTM(d, d, c.num_lstm_layers, batch_first=True,
                                dropout=c.dropout if c.num_lstm_layers > 1 else 0)

        # Decoder path (uses zero future covariates → single-feature VSN over time index)
        self.dec_vsn  = _VariableSelectionNetwork(1, d, c.dropout)
        self.dec_lstm = nn.LSTM(d, d, c.num_lstm_layers, batch_first=True,
                                dropout=c.dropout if c.num_lstm_layers > 1 else 0)

        # Enrichment + attention
        self.enrich_grn = _GRN(d, dropout=c.dropout)
        self.attn       = _TemporalSelfAttention(d, c.num_heads, c.dropout)
        self.attn_gate  = _GLU(d, d)
        self.attn_norm  = nn.LayerNorm(d)
        self.ff_grn     = _GRN(d, dropout=c.dropout)
        self.ff_gate    = _GLU(d, d)
        self.ff_norm    = nn.LayerNorm(d)

        # Output projection
        self.head = nn.Linear(d, c.n_targets)

        # Causal mask (H×H)
        mask = torch.tril(torch.ones(c.horizon, c.horizon)).bool()
        self.register_buffer("causal_mask", mask)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> TSOutput:
        # x: (B, L, F)
        B, L, F = x.shape
        c = self.cfg

        # --- Encoder ---
        enc_in   = self.enc_vsn(x)                         # (B, L, d)
        enc_out, (h, cell) = self.enc_lstm(enc_in)

        # --- Decoder: placeholder future features (time step index, normalised) ---
        t_idx = torch.arange(c.horizon, device=x.device).float() / c.horizon
        dec_raw = t_idx.view(1, c.horizon, 1).expand(B, -1, -1)  # (B,H,1)
        dec_in  = self.dec_vsn(dec_raw)                           # (B,H,d)
        dec_out, _ = self.dec_lstm(dec_in, (h, cell))             # (B,H,d)

        # --- Static enrichment (no static covariates: identity) ---
        enriched = self.enrich_grn(dec_out)                       # (B,H,d)

        # --- Temporal self-attention (causal) ---
        attn_out  = self.attn(enriched, self.causal_mask)
        attn_out  = self.attn_gate(attn_out)
        attn_out  = self.attn_norm(attn_out + enriched)

        # --- Position-wise GRN ---
        ff_out  = self.ff_grn(attn_out)
        ff_out  = self.ff_gate(ff_out)
        ff_out  = self.ff_norm(ff_out + attn_out)

        # --- Output ---
        pred = self.head(ff_out)                                   # (B,H,T)
        return TSOutput(y_hat=pred, extras={"attn_out": attn_out.detach()})
