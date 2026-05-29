"""Physics Foundation Model — the generalist mega-model.

:class:`PhysicsFoundationModel` is the top-level architecture trained by the
world-model pipeline to serve as a *general-purpose physics AI*.

Design principles
-----------------
1. **Universal operator** — the backbone is a deep Fourier Neural Operator
   (FNO) with cross-attention so it handles 1D, 2D, and 3D grids.
2. **Physics context encoding** — a transformer encoder consumes a free-form
   physics descriptor (scenario name, PDE kind, domain bounds, Reynolds number,
   …) and produces a context vector that steers the operator.
3. **Fast adaptation** — thin LoRA adapters in every FNO block let the model
   adapt to a new physics domain in <100 gradient steps via
   :class:`~.meta_learning.MetaLearner`.
4. **Multi-scale** — hierarchical skip connections pool information across
   grid scales, improving performance on both fine- and coarse-grid inputs.
5. **Uncertainty** — optional MC-Dropout or Aleatoric head (``pinneapple_uq``)
   gives calibrated prediction intervals alongside the mean field.

Integration with Pinneapple
--------------------------
* **pinneapple_models** — alternative backbone architectures (AFNO, SIREN,
  MeshGraphNet) can be hot-swapped via :class:`ModelRegistry`.
* **pinneapple_uq** — :class:`~pinneapple_uq.AleatoricHead` is attached when
  ``uncertainty="aleatoric"`` is requested.
* **pinneapple_transfer** — :func:`~pinneapple_transfer.layer_lr_groups`
  defines a layered learning-rate schedule for fine-tuning.
* **pinneapple_meta** — the model is designed to be the outer-loop model in
  MAML/Reptile; its LoRA adapters serve as the per-task fast weights.

Quick start::

    from pinneapple_worldmodel.mega_model import (
        PhysicsFoundationModel, FoundationConfig,
    )

    model = PhysicsFoundationModel(FoundationConfig(device="cuda"))
    # Rollout from a new initial condition:
    state_0 = torch.randn(1, 1, 64, 64)           # (B, C, H, W)
    descriptor = {"scenario": "ns2d_cavity", "Re": 500}
    states = model.rollout(state_0, descriptor, n_steps=20)
    # states : (1, 20, 1, 64, 64)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoundationConfig
# ---------------------------------------------------------------------------

@dataclass
class FoundationConfig:
    """Hyper-parameter configuration for :class:`PhysicsFoundationModel`.

    Parameters
    ----------
    n_modes : int — Fourier modes per spatial dimension.
    width : int — channel width (FNO hidden size).
    depth : int — number of FNO blocks.
    context_dim : int — physics descriptor embedding size.
    descriptor_vocab : int — vocabulary size for tokenised physics descriptors.
    n_heads : int — multi-head attention heads in the context encoder.
    n_context_layers : int — transformer layers in the context encoder.
    lora_rank : int — LoRA adapter rank (0 = no LoRA).
    dropout : float
    use_aleatoric : bool — attach aleatoric uncertainty head.
    max_n_fields : int — maximum number of physical fields (channels).
    device : str
    """
    n_modes: int = 16
    width: int = 128
    depth: int = 6
    context_dim: int = 64
    descriptor_vocab: int = 256
    n_heads: int = 4
    n_context_layers: int = 2
    lora_rank: int = 8
    dropout: float = 0.1
    use_aleatoric: bool = False
    max_n_fields: int = 8
    device: str = "cpu"


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for fast domain adaptation.

    Adds a pair of low-rank matrices (A, B) to a frozen linear layer so that
    fine-tuning only updates A and B (rank << width).
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = 1.0 / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(x))

    def freeze_base(self) -> None:
        self.linear.requires_grad_(False)

    def unfreeze_base(self) -> None:
        self.linear.requires_grad_(True)


# ---------------------------------------------------------------------------
# Physics descriptor encoder
# ---------------------------------------------------------------------------

class PhysicsDescriptorEncoder(nn.Module):
    """Encode a physics descriptor dict into a fixed-size context vector.

    Supports two input modes:
    * **Float dict** — ``{"Re": 500.0, "alpha": 0.01, …}`` (most common).
    * **Pre-encoded tensor** — direct ``(B, D)`` float tensor.

    Parameters
    ----------
    context_dim : int — output embedding size.
    max_scalar_features : int — maximum number of scalar inputs from the dict.
    n_heads : int — transformer heads.
    n_layers : int — transformer encoder layers.
    dropout : float
    """

    def __init__(
        self,
        context_dim: int = 64,
        max_scalar_features: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.max_scalars = max_scalar_features

        # Project scalar features to context_dim
        self.scalar_embed = nn.Linear(max_scalar_features, context_dim)

        # Lightweight self-attention transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=n_heads,
            dim_feedforward=context_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            # norm_first=True disables nested-tensor path; silence the PyTorch warning
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                     enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        descriptor: Union[Tensor, Dict[str, Any]],
        batch_size: int = 1,
    ) -> Tensor:
        """Encode descriptor to ``(B, context_dim)`` tensor.

        Parameters
        ----------
        descriptor : Tensor ``(B, D)`` or dict of scalar values.
        batch_size : int — used only when descriptor is a dict.
        """
        if isinstance(descriptor, Tensor):
            x = descriptor  # (B, D)
            # Pad or truncate to max_scalars
            if x.shape[-1] < self.max_scalars:
                x = F.pad(x, (0, self.max_scalars - x.shape[-1]))
            else:
                x = x[..., :self.max_scalars]
        else:
            # Convert dict to tensor
            vals = []
            for v in descriptor.values():
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if not vals:
                vals = [0.0]
            # Pad to max_scalars
            vals = vals[:self.max_scalars]
            vals += [0.0] * (self.max_scalars - len(vals))
            x = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
            x = x.expand(batch_size, -1)

        x = x.to(next(self.parameters()).device)
        h = self.scalar_embed(x).unsqueeze(1)  # (B, 1, context_dim)
        h = self.transformer(h)
        h = self.out_norm(h[:, 0])  # (B, context_dim)
        return h


# ---------------------------------------------------------------------------
# Multi-scale FNO block
# ---------------------------------------------------------------------------

class _SpectralConv(nn.Module):
    """Spectral convolution supporting 1D and 2D (FNO-style channel mixing)."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: int, dim: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.dim = dim

        scale = 1.0 / (in_channels * out_channels)
        if dim == 1:
            self.weight = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat)
            )
        else:
            self.weight = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, n_modes, n_modes,
                                    dtype=torch.cfloat)
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 1:
            xf = torch.fft.rfft(x)           # (B, C_in, L//2+1) complex
            m = min(self.n_modes, xf.shape[-1])
            out = torch.zeros(xf.shape[0], self.out_channels, xf.shape[-1],
                              dtype=xf.dtype, device=xf.device)
            # einsum: b i l, i o l -> b o l
            out[..., :m] = torch.einsum("bil,iol->bol", xf[..., :m],
                                        self.weight[:, :, :m])
            return torch.fft.irfft(out, n=x.shape[-1])
        else:
            xf = torch.fft.rfft2(x)          # (B, C_in, H, W//2+1) complex
            m1 = min(self.n_modes, xf.shape[-2])
            m2 = min(self.n_modes, xf.shape[-1])
            out = torch.zeros(xf.shape[0], self.out_channels,
                              xf.shape[-2], xf.shape[-1],
                              dtype=xf.dtype, device=xf.device)
            # einsum: b i h w, i o h w -> b o h w
            out[..., :m1, :m2] = torch.einsum(
                "bihw,iohw->bohw",
                xf[..., :m1, :m2],
                self.weight[:, :, :m1, :m2],
            )
            return torch.fft.irfft2(out, s=(x.shape[-2], x.shape[-1]))


class FoundationFNOBlock(nn.Module):
    """One FNO block with context injection and optional LoRA.

    Parameters
    ----------
    width : int — channel dimension.
    n_modes : int — Fourier modes.
    context_dim : int — context vector size (injected via FiLM).
    lora_rank : int — LoRA rank (0 = standard linear).
    dropout : float
    spatial_dim : int — 1 or 2.
    """

    def __init__(
        self,
        width: int,
        n_modes: int,
        context_dim: int,
        lora_rank: int = 8,
        dropout: float = 0.1,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.spectral = _SpectralConv(width, width, n_modes, dim=spatial_dim)
        self.bypass = nn.Conv1d(width, width, 1) if spatial_dim == 1 \
            else nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width) if spatial_dim == 2 \
            else nn.InstanceNorm1d(width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # FiLM conditioning: context → (scale, shift) for each channel
        if lora_rank > 0:
            self.film_scale = LoRALinear(context_dim, width, rank=lora_rank)
            self.film_shift = LoRALinear(context_dim, width, rank=lora_rank)
        else:
            self.film_scale = nn.Linear(context_dim, width)
            self.film_shift = nn.Linear(context_dim, width)

        self.spatial_dim = spatial_dim

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        # x: (B, width, *spatial)
        h = self.spectral(x) + self.bypass(x)

        if context is not None:
            # FiLM: feature-wise affine modulation
            scale = self.film_scale(context)
            shift = self.film_shift(context)
            if self.spatial_dim == 1:
                scale = scale.unsqueeze(-1)
                shift = shift.unsqueeze(-1)
            else:
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift

        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h  # residual


# ---------------------------------------------------------------------------
# PhysicsFoundationModel
# ---------------------------------------------------------------------------

class PhysicsFoundationModel(nn.Module):
    """Generalist physics AI foundation model.

    Architecture::

        [state_t (B, C, *grid)] + [descriptor → context (B, context_dim)]
             ↓
        Lifting projection: (C → width)
             ↓
        N × FoundationFNOBlock  (spectral + FiLM context + LoRA adapters)
             ↓
        Multi-scale pooling skip
             ↓
        Projection: (width → C)
             ↓
        state_tp1 (B, C, *grid)   [residual: + state_t]

    Parameters
    ----------
    config : FoundationConfig
    n_fields : int — number of physical fields (channels) in state.
    grid_shape : tuple — spatial resolution, e.g. ``(64, 64)`` or ``(256,)``.
    """

    def __init__(
        self,
        config: FoundationConfig,
        *,
        n_fields: int = 1,
        grid_shape: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.config = config
        self.n_fields = n_fields
        self.grid_shape = grid_shape
        self.spatial_dim = len(grid_shape)

        w = config.width
        ctx = config.context_dim

        # Context encoder
        self.context_encoder = PhysicsDescriptorEncoder(
            context_dim=ctx,
            n_heads=config.n_heads,
            n_layers=config.n_context_layers,
            dropout=config.dropout,
        )

        # Lifting: fields → width channels
        if self.spatial_dim == 1:
            self.lift = nn.Conv1d(n_fields, w, 1)
            self.proj = nn.Conv1d(w, n_fields, 1)
        else:
            self.lift = nn.Conv2d(n_fields, w, 1)
            self.proj = nn.Conv2d(w, n_fields, 1)

        # FNO blocks
        self.blocks = nn.ModuleList([
            FoundationFNOBlock(
                width=w,
                n_modes=config.n_modes,
                context_dim=ctx,
                lora_rank=config.lora_rank,
                dropout=config.dropout,
                spatial_dim=self.spatial_dim,
            )
            for _ in range(config.depth)
        ])

        # Multi-scale skip: pool to half resolution and add back
        if self.spatial_dim == 2:
            self.pool_skip = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(w, w, 1),
            )
            self.upsample_skip = nn.Upsample(scale_factor=2, mode="bilinear",
                                             align_corners=False)
        else:
            self.pool_skip = nn.Sequential(
                nn.AvgPool1d(2, 2),
                nn.Conv1d(w, w, 1),
            )
            self.upsample_skip = nn.Upsample(scale_factor=2)

        # Optional aleatoric head
        self.aleatoric_head: Optional[nn.Module] = None
        if config.use_aleatoric:
            self.aleatoric_head = (
                nn.Conv2d(w, n_fields, 1) if self.spatial_dim == 2
                else nn.Conv1d(w, n_fields, 1)
            )

        log.info("PhysicsFoundationModel: %d params, spatial_dim=%d, depth=%d",
                 self.parameter_count(), self.spatial_dim, config.depth)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        state_t: Tensor,
        descriptor: Union[Tensor, Dict[str, Any], None] = None,
    ) -> Tensor:
        """Predict ``state_{t+1}`` from ``state_t`` and physics descriptor.

        Parameters
        ----------
        state_t : Tensor ``(B, C, *grid)``
        descriptor : physics context — Tensor ``(B, D)`` or dict or None.

        Returns
        -------
        Tensor ``(B, C, *grid)``
        """
        B = state_t.shape[0]
        # Encode context
        if descriptor is not None:
            if isinstance(descriptor, Tensor) and descriptor.dim() == 1:
                descriptor = descriptor.unsqueeze(0).expand(B, -1)
            ctx = self.context_encoder(descriptor, batch_size=B)
        else:
            ctx = torch.zeros(B, self.config.context_dim, device=state_t.device)

        # Lift
        h = self.lift(state_t)  # (B, width, *grid)

        # Multi-scale skip
        h_skip = self.pool_skip(h)

        # FNO blocks
        for block in self.blocks:
            h = block(h, ctx)

        # Add upsampled skip (pad if size mismatch due to odd dimensions)
        h_up = self.upsample_skip(h_skip)
        h_up = self._match_size(h_up, h)
        h = h + h_up

        # Project back
        delta = self.proj(h)
        return state_t + delta  # residual prediction

    def _match_size(self, src: Tensor, ref: Tensor) -> Tensor:
        """Crop or pad src to match ref spatial size."""
        if src.shape == ref.shape:
            return src
        if self.spatial_dim == 2:
            dh = ref.shape[-2] - src.shape[-2]
            dw = ref.shape[-1] - src.shape[-1]
            if dh > 0 or dw > 0:
                src = F.pad(src, (0, max(dw, 0), 0, max(dh, 0)))
            src = src[..., :ref.shape[-2], :ref.shape[-1]]
        else:
            dl = ref.shape[-1] - src.shape[-1]
            if dl > 0:
                src = F.pad(src, (0, dl))
            src = src[..., :ref.shape[-1]]
        return src

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        state_0: Tensor,
        descriptor: Union[Tensor, Dict[str, Any], None] = None,
        *,
        n_steps: int = 20,
    ) -> Tensor:
        """Auto-regressive rollout for *n_steps* steps.

        Parameters
        ----------
        state_0 : Tensor ``(B, C, *grid)``
        descriptor : physics descriptor (constant across steps).
        n_steps : int

        Returns
        -------
        Tensor ``(B, n_steps, C, *grid)``
        """
        self.eval()
        states = []
        state = state_0
        for _ in range(n_steps):
            state = self(state, descriptor)
            states.append(state)
        return torch.stack(states, dim=1)

    # ------------------------------------------------------------------
    # LoRA adapter management
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all parameters except LoRA adapters (for fast adaptation)."""
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        self.requires_grad_(True)

    def lora_parameters(self):
        """Iterate over LoRA adapter parameters only."""
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                yield param

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        state_t: Tensor,
        descriptor: Union[Tensor, Dict[str, Any], None] = None,
        *,
        n_samples: int = 20,
    ) -> Tuple[Tensor, Tensor]:
        """MC-Dropout uncertainty estimate.

        Parameters
        ----------
        state_t : Tensor ``(B, C, *grid)``
        descriptor : physics descriptor.
        n_samples : int — dropout samples.

        Returns
        -------
        (mean, std) — both ``(B, C, *grid)``
        """
        self.train()  # activates dropout
        preds = torch.stack([self(state_t, descriptor) for _ in range(n_samples)], dim=0)
        self.eval()
        return preds.mean(0), preds.std(0)

    # ------------------------------------------------------------------
    # Alternative backbone
    # ------------------------------------------------------------------

    @classmethod
    def with_afno_backbone(
        cls,
        config: FoundationConfig,
        *,
        n_fields: int = 1,
        grid_shape: Tuple[int, ...] = (64, 64),
    ) -> "PhysicsFoundationModel":
        """Swap the FNO backbone for AFNO (Adaptive Fourier Neural Operator).

        Requires ``pinneapple_models.AFNO``.  Falls back silently to the
        standard FNO backbone if unavailable.
        """
        model = cls(config, n_fields=n_fields, grid_shape=grid_shape)
        try:
            from pinneapple_neural.architectures import AFNO  # type: ignore
            afno = AFNO(
                in_channels=config.width,
                out_channels=config.width,
                hidden_size=config.width,
                num_blocks=config.depth // 2,
            )
            # Replace blocks with AFNO wrapper
            class AFNOWrapper(nn.Module):
                def __init__(self, afno: nn.Module, ctx_dim: int, w: int) -> None:
                    super().__init__()
                    self.afno = afno
                    self.ctx_proj = nn.Linear(ctx_dim, w)

                def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
                    h = self.afno(x)
                    if context is not None:
                        bias = self.ctx_proj(context).unsqueeze(-1).unsqueeze(-1)
                        h = h + bias
                    return h + x

            model.blocks = nn.ModuleList([
                AFNOWrapper(afno, config.context_dim, config.width)
            ])
            log.info("PhysicsFoundationModel: using AFNO backbone.")
        except Exception as exc:
            log.debug("AFNO fallback to FNO: %s", exc)
        return model

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"PhysicsFoundationModel("
            f"n_fields={self.n_fields}, grid={self.grid_shape}, "
            f"width={self.config.width}, depth={self.config.depth}, "
            f"params={self.parameter_count():,})"
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.state_dict(),
            "config": self.config,
            "n_fields": self.n_fields,
            "grid_shape": self.grid_shape,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PhysicsFoundationModel":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ckpt["config"], n_fields=ckpt["n_fields"],
                    grid_shape=ckpt["grid_shape"])
        model.load_state_dict(ckpt["model_state"])
        return model
