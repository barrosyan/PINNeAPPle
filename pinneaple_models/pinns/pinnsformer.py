from __future__ import annotations
"""PINNsFormer transformer-based architecture for physics-informed learning."""
from typing import Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


class PINNsFormer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_len: int = 4096,
        learnable_pos_emb: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.learnable_pos_emb = bool(learnable_pos_emb)

        if self.seq_len > self.max_len:
            raise ValueError(f"seq_len={self.seq_len} > max_len={self.max_len}")

        self.in_proj = nn.Linear(self.in_dim, self.d_model)

        if self.learnable_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.register_buffer("pos_emb", self._sinusoidal_pe(self.max_len, self.d_model), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.out_proj = nn.Linear(self.d_model, self.out_dim)

        self.register_buffer("_attn_mask", None, persistent=False)
        self.register_buffer("_key_padding_mask", None, persistent=False)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def set_masks(
        self,
        *,
        attn_mask: Optional[Tensor] = None,         # (T,T)
        key_padding_mask: Optional[Tensor] = None,  # (B,T) True=PAD
    ) -> None:
        self._attn_mask = attn_mask
        self._key_padding_mask = key_padding_mask

    def forward(self, *inputs: Tensor) -> Tensor:
        if len(inputs) != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} inputs, got {len(inputs)}")

        cols = []
        for t in inputs:
            if t.ndim == 1:
                t = t.unsqueeze(1)
            if t.ndim != 2 or t.shape[1] != 1:
                raise ValueError(f"Each input must be (N,1) (or (N,)), got {tuple(t.shape)}")
            cols.append(t)

        x = torch.cat(cols, dim=1)  # (N, in_dim)
        N = x.shape[0]
        T = self.seq_len
        if N % T != 0:
            raise ValueError(f"N={N} must be multiple of seq_len(T)={T} to reshape into sequences.")

        B = N // T
        x_seq = x.view(B, T, self.in_dim)  # (B,T,in_dim)

        h = self.in_proj(x_seq)            # (B,T,D)
        pe = self.pos_emb[:, :T, :].to(device=h.device, dtype=h.dtype)
        h = h + pe

        h = self.enc(h, mask=self._attn_mask, src_key_padding_mask=self._key_padding_mask)  # (B,T,D)
        y_seq = self.out_proj(h)            # (B,T,out_dim)

        y = y_seq.reshape(B * T, self.out_dim)  # (N,out_dim)
        return y
