"""Physics World Model architecture.

:class:`PhysicsWorldModel` is an FNO-based model that maps a current field
snapshot plus physics context to the next snapshot:

    f_θ: (state_t, context) → state_{t+1}

Architecture
------------
::

    context_enc: MLP(context_dim → embed_dim)          ← physics parameters
    state_aug:   concat(state_t, context_grid)          ← broadcast embed over grid
    fno_blocks:  FourierLayer × depth                   ← spectral + local mixing
    residual:    state_t + W_out(fno_out)               ← learn the delta only

The residual formulation (predict *increment* rather than full next state) is
critical for stability across many rollout steps.

For 1-D problems the Fourier layers use ``fft`` / ``irfft``; for 2-D they
use ``rfft2`` / ``irfft2``.  The spatial dimension is inferred from the grid
shape of the first batch.

Quick start::

    from pinneapple_worldmodel import PhysicsWorldModel, WorldModelConfig

    cfg   = WorldModelConfig(n_modes=16, width=64, depth=4, context_dim=8)
    model = PhysicsWorldModel(cfg, n_fields=1, grid_shape=(64, 64))
    # model(state_t, context) → state_tp1, same shape as state_t
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    """Hyperparameters for :class:`PhysicsWorldModel`.

    Parameters
    ----------
    n_modes : int
        Number of Fourier modes retained (per spatial dimension).
    width : int
        Internal channel width of the FNO blocks.
    depth : int
        Number of Fourier / residual blocks.
    context_dim : int
        Dimensionality of the context vector (PDE params + one-hot kind).
        Set to 0 to disable context conditioning.
    embed_dim : int
        Dimensionality of the context MLP output (added as extra channels).
    activation : str
        Non-linearity: ``"gelu"`` or ``"relu"``.
    dropout : float
        Dropout rate applied after each FNO block (0 = disabled).
    rollout_steps : int
        Number of auto-regressive steps used during training loss.
        1 = next-step only; >1 = multi-step rollout loss.
    """
    n_modes: int = 16
    width: int = 64
    depth: int = 4
    context_dim: int = 8
    embed_dim: int = 16
    activation: str = "gelu"
    dropout: float = 0.0
    rollout_steps: int = 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ContextEncoder(nn.Module):
    """Small MLP projecting context → spatial embedding."""

    def __init__(self, context_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, ctx: Tensor) -> Tensor:
        """ctx : (B, context_dim) → (B, embed_dim)."""
        return self.net(ctx)


class _SpectralConv1d(nn.Module):
    """Fourier layer for 1-D fields."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int) -> None:
        super().__init__()
        self.n_modes = n_modes
        scale = 1.0 / math.sqrt(in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, Nx) → (B, out_ch, Nx)."""
        B, C, Nx = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        modes = min(self.n_modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.weight.shape[1], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes] = torch.einsum(
            "bci,coi->boi", x_ft[:, :, :modes], self.weight[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=Nx, dim=-1)


class _SpectralConv2d(nn.Module):
    """Fourier layer for 2-D fields."""

    def __init__(self, in_ch: int, out_ch: int, n_modes: int) -> None:
        super().__init__()
        self.n_modes = n_modes
        scale = 1.0 / math.sqrt(in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, n_modes, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, Nx, Ny) → (B, out_ch, Nx, Ny)."""
        B, C, Nx, Ny = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        mx = min(self.n_modes, x_ft.shape[-2])
        my = min(self.n_modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.weight.shape[1], x_ft.shape[-2], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bcij,coij->boij",
            x_ft[:, :, :mx, :my],
            self.weight[:, :, :mx, :my],
        )
        return torch.fft.irfft2(out_ft, s=(Nx, Ny), dim=(-2, -1))


class _FNOBlock(nn.Module):
    """One FNO residual block: spectral conv + pointwise conv + activation."""

    def __init__(
        self,
        width: int,
        n_modes: int,
        spatial_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        SpConv = _SpectralConv1d if spatial_dim == 1 else _SpectralConv2d
        self.sp_conv = SpConv(width, width, n_modes)
        self.pw_conv = nn.Conv1d(width, width, 1) if spatial_dim == 1 \
                  else nn.Conv2d(width, width, 1)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.act(self.sp_conv(x) + self.pw_conv(x)))


# ---------------------------------------------------------------------------
# PhysicsWorldModel
# ---------------------------------------------------------------------------

class PhysicsWorldModel(nn.Module):
    """FNO-based physics world model: (state_t, context) → state_{t+1}.

    Parameters
    ----------
    config : WorldModelConfig
    n_fields : int — number of field channels (e.g. 1 for temperature, 3 for u/v/p).
    grid_shape : tuple — spatial grid, e.g. ``(64, 64)`` for 2-D.
    """

    def __init__(
        self,
        config: WorldModelConfig,
        *,
        n_fields: int,
        grid_shape: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.config = config
        self.n_fields = n_fields
        self.grid_shape = grid_shape
        self.spatial_dim = len(grid_shape)

        embed_dim = config.embed_dim if config.context_dim > 0 else 0
        in_ch = n_fields + embed_dim

        # Context encoder
        self.ctx_enc: Optional[nn.Module] = None
        if config.context_dim > 0:
            self.ctx_enc = _ContextEncoder(config.context_dim, embed_dim)

        # Lifting projection: in_ch → width
        ConvLift = nn.Conv1d if self.spatial_dim == 1 else nn.Conv2d
        self.lift = ConvLift(in_ch, config.width, 1)

        # FNO blocks
        self.blocks = nn.ModuleList([
            _FNOBlock(config.width, config.n_modes, self.spatial_dim,
                      config.activation, config.dropout)
            for _ in range(config.depth)
        ])

        # Projection back to field space
        ConvProj = nn.Conv1d if self.spatial_dim == 1 else nn.Conv2d
        self.proj = nn.Sequential(
            ConvProj(config.width, config.width // 2, 1),
            nn.GELU(),
            ConvProj(config.width // 2, n_fields, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        state_t: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict next state.

        Parameters
        ----------
        state_t : Tensor ``(B, C, *grid)``
            Current field snapshot.
        context : Tensor ``(B, context_dim)`` or None
            PDE parameter context.  If None, no conditioning is applied.

        Returns
        -------
        Tensor ``(B, C, *grid)`` — predicted state_{t+1}.
        """
        # Augment state with context broadcast over grid
        x = state_t
        if self.ctx_enc is not None and context is not None:
            embed = self.ctx_enc(context)              # (B, embed_dim)
            grid_embed = self._broadcast_embed(embed, state_t.shape)
            x = torch.cat([state_t, grid_embed], dim=1)

        # Lift
        x = self.lift(x)

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Project → delta, then residual
        delta = self.proj(x)
        return state_t + delta

    def rollout(
        self,
        state_0: Tensor,
        context: Optional[Tensor],
        n_steps: int,
    ) -> Tensor:
        """Auto-regressive rollout.

        Parameters
        ----------
        state_0 : Tensor ``(B, C, *grid)``
        context : Tensor ``(B, context_dim)`` or None
        n_steps : int — how many steps to unroll.

        Returns
        -------
        Tensor ``(B, n_steps, C, *grid)`` — all predicted states (excluding t=0).
        """
        states = []
        state = state_0
        for _ in range(n_steps):
            state = self(state, context)
            states.append(state)
        return torch.stack(states, dim=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _broadcast_embed(self, embed: Tensor, state_shape: Tuple) -> Tensor:
        """Expand (B, embed_dim) → (B, embed_dim, *grid_shape)."""
        B, D = embed.shape
        view = embed
        for _ in range(self.spatial_dim):
            view = view.unsqueeze(-1)
        expand_shape = (B, D, *self.grid_shape)
        return view.expand(expand_shape)

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"PhysicsWorldModel("
            f"n_fields={self.n_fields}, grid={self.grid_shape}, "
            f"depth={self.config.depth}, width={self.config.width}, "
            f"params={self.parameter_count():,})"
        )
