from __future__ import annotations
"""Fourier neural operator for global spectral learning (1D and 2D)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NeuralOperatorBase, OperatorOutput


class _SpectralConv(nn.Module):
    """
    1D Spectral Convolution:
      - FFT on last dimension
      - linear transform on low-frequency modes
      - iFFT back

    x: (B, in_c, L) -> (B, out_c, L)
    """

    def __init__(self, in_c: int, out_c: int, modes: int):
        super().__init__()
        self.in_c = int(in_c)
        self.out_c = int(out_c)
        self.modes = int(modes)

        # Complex weights for low-frequency modes: (in_c, out_c, modes)
        # Scale is a common heuristic; other init strategies are also used.
        scale = 1.0 / max(1, (in_c * out_c))
        self.weights = nn.Parameter(scale * torch.randn(in_c, out_c, self.modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_c, L)
        B, C, L = x.shape
        if C != self.in_c:
            raise ValueError(f"_SpectralConv expected in_c={self.in_c}, got C={C}")

        # rFFT: (B, in_c, Lf) where Lf = L//2 + 1
        x_ft = torch.fft.rfft(x, dim=-1)
        _, _, Lf = x_ft.shape

        # In case modes > Lf (small L), clamp
        m = min(self.modes, Lf)

        # IMPORTANT: out_ft must have out_c channels (not in_c)
        out_ft = torch.zeros(
            (B, self.out_c, Lf),
            device=x.device,
            dtype=torch.cfloat,
        )

        # Multiply low modes: (B,in_c,m) x (in_c,out_c,m) -> (B,out_c,m)
        out_ft[:, :, :m] = torch.einsum("bcm,com->bom", x_ft[:, :, :m], self.weights[:, :, :m])

        # iFFT back to (B, out_c, L)
        y = torch.fft.irfft(out_ft, n=L, dim=-1)
        return y


class FourierNeuralOperator(NeuralOperatorBase):
    """
    FNO-1D (MVP) - extendable to 2D/3D by changing spectral conv + projections.

    Core block:
      x <- GELU( SpectralConv(x) + 1x1Conv(x) )

    Optional:
      - use_grid=True: concatenate 1D coordinate grid as an extra input channel
        (common in FNO literature for better generalization / positional info).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: int = 16,
        layers: int = 4,
        *,
        use_grid: bool = False,
    ):
        super().__init__()
        self.use_grid = bool(use_grid)

        in_c_eff = int(in_channels) + (1 if self.use_grid else 0)
        self.in_proj = nn.Conv1d(in_c_eff, int(width), kernel_size=1)

        self.convs = nn.ModuleList([_SpectralConv(int(width), int(width), int(modes)) for _ in range(int(layers))])
        self.ws = nn.ModuleList([nn.Conv1d(int(width), int(width), kernel_size=1) for _ in range(int(layers))])

        self.out_proj = nn.Conv1d(int(width), int(out_channels), kernel_size=1)

    @staticmethod
    def _make_grid_1d(u: torch.Tensor) -> torch.Tensor:
        """
        Create a normalized 1D grid in [0,1], shaped as (B,1,L), matching u.device/u.dtype.
        """
        B, _, L = u.shape
        grid = torch.linspace(0.0, 1.0, L, device=u.device, dtype=u.dtype).view(1, 1, L)
        return grid.repeat(B, 1, 1)

    def forward(
        self,
        u: torch.Tensor,  # (B, C, L)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        x = u
        if self.use_grid:
            grid = self._make_grid_1d(u)
            x = torch.cat([x, grid], dim=1)  # (B, C+1, L)

        x = self.in_proj(x)

        for sc, w in zip(self.convs, self.ws):
            x = F.gelu(sc(x) + w(x))

        y = self.out_proj(x)

        losses = {}
        if return_loss and y_true is not None:
            mse = self.mse(y, y_true)
            losses["mse"] = mse
            losses["total"] = mse
        elif return_loss:
            # If user asked for loss but didn't provide y_true, keep total=0 for compatibility
            losses["total"] = torch.tensor(0.0, device=y.device)

        return OperatorOutput(y=y, losses=losses, extras={"use_grid": self.use_grid})


# ─────────────────────────────────────────────────────────────────────────────
# FNO-2D
# ─────────────────────────────────────────────────────────────────────────────

class _SpectralConv2d(nn.Module):
    """2D Spectral Convolution: (B, in_c, H, W) → (B, out_c, H, W).

    Uses ``rfft2`` so only positive frequencies along the last dimension are
    stored.  Two weight tensors cover the positive and negative low-frequency
    modes in the penultimate dimension.
    """

    def __init__(self, in_c: int, out_c: int, modes1: int, modes2: int):
        super().__init__()
        self.in_c   = int(in_c)
        self.out_c  = int(out_c)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 1.0 / max(1, in_c * out_c)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_c, out_c, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_c, out_c, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # rfft2 → (B, C, H, W//2+1)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        _, _, H_f, W_f = x_ft.shape

        m1 = min(self.modes1, H_f // 2)
        m2 = min(self.modes2, W_f)

        out_ft = torch.zeros(B, self.out_c, H_f, W_f, device=x.device, dtype=torch.cfloat)

        # Positive low-frequency corner (top-left in H dimension)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bihw,iohw->bohw",
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        # Negative low-frequency corner (bottom-left in H dimension)
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bihw,iohw->bohw",
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))


class FNO2d(NeuralOperatorBase):
    """FNO-2D: Fourier Neural Operator for 2D spatial fields.

    Input/output shape: ``(B, in_channels, H, W)`` → ``(B, out_channels, H, W)``.

    Parameters
    ----------
    in_channels : int
        Input channel count (e.g. 1 for a scalar field).
    out_channels : int
        Output channel count.
    width : int
        Lifted channel width inside the FNO blocks.
    modes1 : int
        Number of Fourier modes to retain in the H direction.
    modes2 : int
        Number of Fourier modes to retain in the W direction.
    layers : int
        Number of Fourier + bypass layers.
    use_grid : bool
        Concatenate 2D normalised coordinate grids (x ∈ [0,1], y ∈ [0,1]) as
        two extra input channels.  Recommended for non-periodic domains.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        width:  int = 32,
        modes1: int = 12,
        modes2: int = 12,
        layers: int = 4,
        *,
        use_grid: bool = True,
    ):
        super().__init__()
        self.use_grid = bool(use_grid)

        in_c_eff = int(in_channels) + (2 if self.use_grid else 0)
        self.in_proj = nn.Conv2d(in_c_eff, int(width), kernel_size=1)

        self.convs = nn.ModuleList([
            _SpectralConv2d(int(width), int(width), int(modes1), int(modes2))
            for _ in range(int(layers))
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(int(width), int(width), kernel_size=1)
            for _ in range(int(layers))
        ])
        self.out_proj = nn.Conv2d(int(width), int(out_channels), kernel_size=1)

    @staticmethod
    def _make_grid_2d(u: torch.Tensor) -> torch.Tensor:
        """Normalised 2D grid (B, 2, H, W) with values in [0, 1]."""
        B, _, H, W = u.shape
        gx = torch.linspace(0.0, 1.0, H, device=u.device, dtype=u.dtype)
        gy = torch.linspace(0.0, 1.0, W, device=u.device, dtype=u.dtype)
        GX, GY = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack([GX, GY], dim=0).unsqueeze(0)  # (1, 2, H, W)
        return grid.expand(B, -1, -1, -1)

    def forward(
        self,
        u: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        x = u
        if self.use_grid:
            grid = self._make_grid_2d(u)
            x = torch.cat([x, grid], dim=1)

        x = self.in_proj(x)
        for sc, w in zip(self.convs, self.ws):
            x = F.gelu(sc(x) + w(x))
        y = self.out_proj(x)

        losses = {}
        if return_loss and y_true is not None:
            mse = self.mse(y, y_true)
            losses["mse"] = mse
            losses["total"] = mse
        elif return_loss:
            losses["total"] = torch.tensor(0.0, device=y.device)

        return OperatorOutput(y=y, losses=losses, extras={"use_grid": self.use_grid})


# ─────────────────────────────────────────────────────────────────────────────
# MLP + FNO Hybrid Surrogate
# ─────────────────────────────────────────────────────────────────────────────

class MLPFNOSurrogate(nn.Module):
    """Hybrid FNO backbone + MLP decoder for scattered-point evaluation.

    The FNO backbone processes a structured 2D context grid (e.g. boundary
    conditions, parameter fields) and produces a global feature map.  An MLP
    decoder then evaluates at arbitrary query coordinates by:

      1. Bilinear-interpolating the FNO feature map at each query location.
      2. Concatenating the sampled features with the query coordinates.
      3. Passing through a small MLP to produce the final prediction.

    This enables global spectral context (FNO) with point-wise evaluation at
    scattered locations, making it autograd-compatible for PINN losses.

    Parameters
    ----------
    in_channels : int
        Context grid input channels (e.g. encoded boundary conditions).
    out_channels : int
        Number of predicted output fields.
    fno_width : int
        Lifted channel width inside FNO blocks.
    fno_modes : int
        Fourier modes retained per spatial dimension.
    fno_layers : int
        Number of FNO Fourier + bypass blocks.
    mlp_hidden : int
        Hidden width of the MLP decoder.
    mlp_layers : int
        Depth of the MLP decoder (must be >= 1).
    coord_dim : int
        Query point dimensionality (default 2 for (x, y)).

    Shapes
    ------
    context_grid  : (B, in_channels, H, W)
    query_coords  : (B, N, coord_dim) or (N, coord_dim) -- in [0, 1]
    output        : (B, N, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fno_width: int = 32,
        fno_modes: int = 12,
        fno_layers: int = 4,
        mlp_hidden: int = 128,
        mlp_layers: int = 3,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.coord_dim = int(coord_dim)

        self._fno = FNO2d(
            in_channels=in_channels,
            out_channels=fno_width,
            width=fno_width,
            modes1=fno_modes,
            modes2=fno_modes,
            layers=fno_layers,
            use_grid=True,
        )

        mlp_in = coord_dim + fno_width
        layers_list: list = []
        prev = mlp_in
        for _ in range(max(mlp_layers - 1, 0)):
            layers_list += [nn.Linear(prev, mlp_hidden), nn.Tanh()]
            prev = mlp_hidden
        layers_list.append(nn.Linear(prev, out_channels))
        self._mlp = nn.Sequential(*layers_list)

    def forward(
        self,
        context_grid: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate predicted fields at scattered query coordinates.

        Parameters
        ----------
        context_grid : (B, in_channels, H, W)
        query_coords : (B, N, coord_dim) or (N, coord_dim) in [0, 1]

        Returns
        -------
        (B, N, out_channels)
        """
        B = context_grid.shape[0]

        fno_out = self._fno(context_grid)
        feat_map = fno_out.y if hasattr(fno_out, "y") else fno_out  # (B, fno_width, H, W)

        if query_coords.dim() == 2:
            query_coords = query_coords.unsqueeze(0).expand(B, -1, -1)

        # grid_sample expects coordinates in [-1, 1]; our coords are in [0, 1]
        grid = query_coords * 2.0 - 1.0  # (B, N, 2)
        grid = grid.unsqueeze(1)          # (B, 1, N, 2)

        sampled = F.grid_sample(
            feat_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(2).permute(0, 2, 1)  # (B, N, fno_width)

        decoder_in = torch.cat([query_coords, sampled], dim=-1)  # (B, N, coord+fno_width)
        return self._mlp(decoder_in)                              # (B, N, out_channels)