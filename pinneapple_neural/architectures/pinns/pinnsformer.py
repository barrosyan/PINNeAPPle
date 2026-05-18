from __future__ import annotations
"""PINNsFormer - Transformer-based architecture for physics-informed learning.

Reference: Zhao et al. (2023) "PINNsFormer: A Transformer-Based Framework
For Physics-Informed Neural Networks" arXiv:2307.11833

Key ideas
---------
- Coordinates (x, t, ...) are treated as a *sequence* over collocation points.
- The model learns spatial/temporal correlations through self-attention.
- A learnable (or sinusoidal) positional embedding is added per timestep/point.
- Output: field values at each point in the sequence.

Standard interface
------------------
``forward(x)`` where ``x`` is **either**:
  - ``(N, in_dim)`` flat batch  — points are split into sequences of length T
  - ``(B, T, in_dim)`` pre-batched sequences

Both cases return ``(N, out_dim)`` flat (or ``(B, T, out_dim)`` if pre-batched).
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput

Tensor = torch.Tensor


class PINNsFormer(PINNBase):
    """
    Transformer-based PINN.

    Parameters
    ----------
    in_dim : int     input coordinate dimension (e.g. 2 for (x,t))
    out_dim : int    output field dimension (e.g. 1 for u)
    seq_len : int    sequence length T; N must be divisible by T when
                     passing flat (N, in_dim) inputs
    d_model : int    transformer embedding dimension
    nhead : int      number of attention heads (must divide d_model)
    num_layers : int number of TransformerEncoder layers
    dim_feedforward : int  FFN hidden dim inside each encoder layer
    dropout : float  dropout rate (use 0 for PINNs in training)
    max_len : int    maximum sequence length for positional embedding
    learnable_pos_emb : bool  use learnable vs sinusoidal positional embedding
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        seq_len: int = 32,
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

        # Input projection: in_dim -> d_model
        self.in_proj = nn.Linear(self.in_dim, self.d_model)

        # Positional embedding
        if self.learnable_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.register_buffer(
                "pos_emb", self._sinusoidal_pe(self.max_len, self.d_model), persistent=False
            )

        # Transformer encoder (Pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,     # (B, T, D) convention
            activation="gelu",
            norm_first=True,      # Pre-LN (more stable)
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        # Output projection: d_model -> out_dim
        self.out_proj = nn.Linear(self.d_model, self.out_dim)

        # Learnable input-scale (Fourier feature variant — keeps coords in good range)
        self.input_scale = nn.Parameter(torch.ones(self.in_dim))

        # inverse_params for inverse PINNs
        self.inverse_params = nn.ParameterDict()

        self.register_buffer("_attn_mask", None, persistent=False)
        self.register_buffer("_key_padding_mask", None, persistent=False)

    # ------------------------------------------------------------------
    # Positional embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def set_masks(
        self,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> None:
        """Set optional attention / key-padding masks for the transformer."""
        self._attn_mask = attn_mask
        self._key_padding_mask = key_padding_mask

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _encode(self, x_seq: Tensor) -> Tensor:
        """
        x_seq : (B, T, in_dim)
        returns: (B, T, out_dim)
        """
        # Scale coordinates (helps when coords have different magnitudes)
        h = x_seq * self.input_scale.unsqueeze(0).unsqueeze(0)

        h = self.in_proj(h)                             # (B, T, d_model)
        T = h.shape[1]
        pe = self.pos_emb[:, :T, :].to(device=h.device, dtype=h.dtype)
        h = h + pe

        h = self.enc(
            h,
            mask=self._attn_mask,
            src_key_padding_mask=self._key_padding_mask,
        )                                               # (B, T, d_model)
        return self.out_proj(h)                         # (B, T, out_dim)

    def forward_tensor(self, x: Tensor) -> Tensor:
        """
        Standard tensor-only forward; used by PINNFactory / pipeline.

        x : (N, in_dim) or (B, T, in_dim)
        returns : (N, out_dim) or (B, T, out_dim)
        """
        if x.ndim == 2:
            # Flat batch: (N, in_dim) -> split into sequences of length T
            N, D = x.shape
            T = self.seq_len
            if N % T != 0:
                raise ValueError(
                    f"Flat input N={N} must be divisible by seq_len={T}. "
                    "Pass pre-batched (B, T, in_dim) or adjust seq_len."
                )
            B = N // T
            y_seq = self._encode(x.view(B, T, D))      # (B, T, out_dim)
            return y_seq.reshape(N, self.out_dim)        # (N, out_dim)

        if x.ndim == 3:
            # Pre-batched: (B, T, in_dim)
            return self._encode(x)                      # (B, T, out_dim)

        raise ValueError(f"PINNsFormer expects 2D or 3D input, got {x.ndim}D.")

    def forward(
        self,
        x: Tensor,
        *,
        physics_fn=None,
        physics_data=None,
    ) -> PINNOutput:
        """
        Standard PINNBase forward.

        x : (N, in_dim) or (B, T, in_dim)
        """
        y = self.forward_tensor(x)

        z = torch.zeros((), device=y.device, dtype=y.dtype)
        losses = {"total": z}

        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get("physics", z)

        return PINNOutput(y=y, losses=losses, extras={})
