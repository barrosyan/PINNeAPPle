from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NeuralOperatorBase, OperatorOutput


MeshMode = Literal["auto", "grid", "points"]


# -------------------------
# Utilities
# -------------------------
def _nd_interpolate(x: torch.Tensor, size: Tuple[int, ...], dim: int) -> torch.Tensor:
    # x: (B,C,*spatial)
    if dim == 1:
        return F.interpolate(x, size=size, mode="linear", align_corners=False)
    if dim == 2:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if dim == 3:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    raise ValueError(f"dim must be 1/2/3, got {dim}")


def _coords_grid(batch: int, spatial: Tuple[int, ...], device, dim: int) -> torch.Tensor:
    # returns (B, dim, *spatial) in [0,1]
    if dim == 1:
        (L,) = spatial
        x = torch.linspace(0, 1, L, device=device)
        grid = x[None, None, :].expand(batch, 1, L)
        return grid
    if dim == 2:
        H, W = spatial
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)[None, ...].expand(batch, 2, H, W)
        return grid
    if dim == 3:
        D, H, W = spatial
        zs = torch.linspace(0, 1, D, device=device)
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=0)[None, ...].expand(batch, 3, D, H, W)
        return grid
    raise ValueError(f"dim must be 1/2/3, got {dim}")


# -------------------------
# SpectralConv ND (grid)
# -------------------------
class SpectralConvND(nn.Module):
    """
    Spectral convolution generalized to 1D/2D/3D (rfftn).
    Applies a learned complex linear map on low-frequency modes.
    """

    def __init__(self, dim: int, in_channels: int, out_channels: int, modes: Tuple[int, ...]):
        super().__init__()
        assert dim in (1, 2, 3)
        assert len(modes) == dim
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # weight: (in, out, *modes, 2) for complex
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, *modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, *spatial)
        B, Cin = x.shape[:2]
        spatial = x.shape[2:]
        if len(spatial) != self.dim:
            raise ValueError(f"Expected {self.dim}D spatial input, got shape {x.shape}")

        x_ft = torch.fft.rfftn(x, dim=tuple(range(2, 2 + self.dim)), norm="ortho")
        # output in Fourier domain
        out_shape = (B, self.out_channels, *spatial[:-1], spatial[-1] // 2 + 1) if self.dim >= 1 else (B, self.out_channels)
        out_ft = torch.zeros(out_shape, dtype=torch.cfloat, device=x.device)

        w = torch.view_as_complex(self.weight)  # (Cin, Cout, *modes)

        # slice low modes
        if self.dim == 1:
            (m1,) = self.modes
            out_ft[:, :, :m1] = torch.einsum("bix,iox->box", x_ft[:, :, :m1], w)
        elif self.dim == 2:
            m1, m2 = self.modes
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w)
        else:
            m1, m2, m3 = self.modes
            out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
                "bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], w
            )

        y = torch.fft.irfftn(out_ft, s=spatial, dim=tuple(range(2, 2 + self.dim)), norm="ortho")
        return y


class OperatorBlockND(nn.Module):
    def __init__(self, dim: int, width: int, modes: Tuple[int, ...]):
        super().__init__()
        self.spec = SpectralConvND(dim, width, width, modes=modes)
        self.pw = nn.Conv1d(width, width, 1) if dim == 1 else (nn.Conv2d(width, width, 1) if dim == 2 else nn.Conv3d(width, width, 1))
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.pw(x)
        y = self.norm(y)
        y = self.act(y)
        return x + y


class DownBlockND(nn.Module):
    def __init__(self, dim: int, in_ch: int, out_ch: int, modes: Tuple[int, ...], depth: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if dim == 1 else (nn.Conv2d(in_ch, out_ch, 1) if dim == 2 else nn.Conv3d(in_ch, out_ch, 1))
        self.ops = nn.Sequential(*[OperatorBlockND(dim, out_ch, modes) for _ in range(depth)])
        # strided conv downsample
        if dim == 1:
            self.down = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)
        elif dim == 2:
            self.down = nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.ops(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlockND(nn.Module):
    def __init__(self, dim: int, in_ch: int, skip_ch: int, out_ch: int, modes: Tuple[int, ...], depth: int):
        super().__init__()
        self.dim = dim
        self.mix = nn.Conv1d(in_ch + skip_ch, out_ch, 1) if dim == 1 else (nn.Conv2d(in_ch + skip_ch, out_ch, 1) if dim == 2 else nn.Conv3d(in_ch + skip_ch, out_ch, 1))
        self.ops = nn.Sequential(*[OperatorBlockND(dim, out_ch, modes) for _ in range(depth)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # upsample to match skip resolution
        target_spatial = skip.shape[2:]
        x = _nd_interpolate(x, size=target_spatial, dim=self.dim)
        x = torch.cat([x, skip], dim=1)
        x = self.mix(x)
        x = self.ops(x)
        return x


class UNONdGrid(nn.Module):
    """
    U-shaped Neural Operator for regular grids (1D/2D/3D).
    Input:  (B, Cin, *spatial)
    Output: (B, Cout, *spatial)
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        levels: int = 3,
        depth_per_level: int = 2,
        modes: Optional[Tuple[int, ...]] = None,
        add_coords: bool = True,
    ):
        super().__init__()
        assert dim in (1, 2, 3)
        self.dim = dim
        self.add_coords = add_coords

        if modes is None:
            # default: conservative low modes
            modes = (12,) * dim
        assert len(modes) == dim

        lift_in = in_channels + (dim if add_coords else 0)
        self.lift = nn.Conv1d(lift_in, width, 1) if dim == 1 else (nn.Conv2d(lift_in, width, 1) if dim == 2 else nn.Conv3d(lift_in, width, 1))

        downs = []
        ch = width
        for _ in range(levels):
            downs.append(DownBlockND(dim, ch, ch * 2, modes=modes, depth=depth_per_level))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        self.bottleneck = nn.Sequential(*[OperatorBlockND(dim, ch, modes=modes) for _ in range(depth_per_level)])

        ups = []
        for _ in range(levels):
            ups.append(UpBlockND(dim, ch, ch // 2, ch // 2, modes=modes, depth=depth_per_level))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.proj = nn.Sequential(
            (nn.Conv1d(width, width, 1) if dim == 1 else (nn.Conv2d(width, width, 1) if dim == 2 else nn.Conv3d(width, width, 1))),
            nn.GELU(),
            (nn.Conv1d(width, out_channels, 1) if dim == 1 else (nn.Conv2d(width, out_channels, 1) if dim == 2 else nn.Conv3d(width, out_channels, 1))),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B,Cin,*spatial)
        if self.add_coords:
            B = u.shape[0]
            spatial = tuple(u.shape[2:])
            grid = _coords_grid(B, spatial, u.device, dim=self.dim)
            u = torch.cat([u, grid], dim=1)

        x = self.lift(u)

        skips: List[torch.Tensor] = []
        for down in self.downs:
            x, s = down(x)
            skips.append(s)

        x = self.bottleneck(x)

        for up in self.ups:
            s = skips.pop()
            x = up(x, s)

        y = self.proj(x)
        return y


# -------------------------
# Point operator blocks (irregular mesh)
# -------------------------
class PointKernelOperator(nn.Module):
    """
    Continuous operator on irregular points using kNN message passing with a learned kernel.
    x:      (B, N, Cin)
    coords: (B, N, dim)
    out:    (B, N, Cout)

    NOTE: Uses cdist (O(N^2)). Good MVP; swap for fast kNN later.
    """

    def __init__(self, dim: int, in_ch: int, out_ch: int, hidden: int = 64, k: int = 16):
        super().__init__()
        self.dim = dim
        self.k = k

        # kernel MLP takes relative coords (dim) + source features (in_ch)
        self.kernel = nn.Sequential(
            nn.Linear(dim + in_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
        )

        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, Cin = x.shape
        # pairwise distances
        d = torch.cdist(coords, coords)  # (B,N,N)
        idx = d.topk(k=min(self.k, N), largest=False).indices  # (B,N,k)

        # gather neighbors
        # coords_j: (B,N,k,dim)
        coords_j = coords[:, None, :, :].expand(B, N, N, self.dim).gather(2, idx[..., None].expand(B, N, idx.shape[-1], self.dim))
        x_j = x[:, None, :, :].expand(B, N, N, Cin).gather(2, idx[..., None].expand(B, N, idx.shape[-1], Cin))

        # relative coords (neighbor - center)
        rel = coords_j - coords[:, :, None, :]  # (B,N,k,dim)

        inp = torch.cat([rel, x_j], dim=-1)  # (B,N,k,dim+Cin)
        msg = self.kernel(inp)  # (B,N,k,Cout)
        out = msg.mean(dim=2) + self.bias  # aggregate
        return out


class PointOperatorBlock(nn.Module):
    def __init__(self, dim: int, width: int, k: int = 16):
        super().__init__()
        self.op = PointKernelOperator(dim, width, width, hidden=max(64, width), k=k)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        y = self.op(x, coords)
        y = self.norm(x + y)
        y = y + self.ff(y)
        return self.norm(y)


class UNONdPoints(nn.Module):
    """
    U-shaped operator for irregular meshes (points).
    This builds a multi-scale pyramid by subsampling points and using kNN operators.
    Input:  x      (B,N,Cin)
            coords (B,N,dim)
    Output: y      (B,N,Cout)
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int = 128,
        levels: int = 3,
        depth_per_level: int = 2,
        k: int = 16,
        pool_ratio: float = 0.5,
    ):
        super().__init__()
        assert dim in (1, 2, 3)
        self.dim = dim
        self.levels = levels
        self.pool_ratio = pool_ratio

        self.lift = nn.Sequential(
            nn.Linear(in_channels + dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

        self.down_ops = nn.ModuleList([
            nn.ModuleList([PointOperatorBlock(dim, width, k=k) for _ in range(depth_per_level)])
            for _ in range(levels)
        ])
        self.up_ops = nn.ModuleList([
            nn.ModuleList([PointOperatorBlock(dim, width, k=k) for _ in range(depth_per_level)])
            for _ in range(levels)
        ])

        self.proj = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, out_channels),
        )

    def _pool(self, x: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # simple random pooling (MVP). Replace with FPS for better quality.
        B, N, C = x.shape
        M = max(2, int(N * self.pool_ratio))
        idx = torch.rand(B, N, device=x.device).argsort(dim=1)[:, :M]  # (B,M)
        coords_m = coords.gather(1, idx[..., None].expand(B, M, self.dim))
        x_m = x.gather(1, idx[..., None].expand(B, M, C))
        return x_m, coords_m, idx

    def _unpool(self, x_coarse: torch.Tensor, coords_coarse: torch.Tensor, coords_fine: torch.Tensor) -> torch.Tensor:
        # interpolate coarse -> fine by nearest neighbor (MVP)
        # d: (B, Nfine, Ncoarse)
        d = torch.cdist(coords_fine, coords_coarse)
        nn_idx = d.argmin(dim=-1)  # (B,Nfine)
        B, Nf = nn_idx.shape
        C = x_coarse.shape[-1]
        out = x_coarse.gather(1, nn_idx[..., None].expand(B, Nf, C))
        return out

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # x: (B,N,Cin), coords: (B,N,dim)
        h = self.lift(torch.cat([x, coords], dim=-1))

        skips: List[Tuple[torch.Tensor, torch.Tensor]] = []
        pyramids: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Down
        for lvl in range(self.levels):
            for op in self.down_ops[lvl]:
                h = op(h, coords)
            skips.append((h, coords))
            h, coords, _ = self._pool(h, coords)  # go coarser
            pyramids.append((h, coords))

        # Up
        for lvl in range(self.levels - 1, -1, -1):
            # target skip
            h_skip, coords_skip = skips[lvl]
            h = self._unpool(h, coords, coords_skip)
            coords = coords_skip
            h = h + h_skip  # skip connection (sum)

            for op in self.up_ops[lvl]:
                h = op(h, coords)

        y = self.proj(h)
        return y


# -------------------------
# Unified wrapper (auto)
# -------------------------
class UniversalUNO(NeuralOperatorBase):
    """
    Unified U-NO module:
    - grid mode: (B,C,*spatial) -> (B,Cout,*spatial)
    - points mode: (B,N,Cin) + coords(B,N,dim) -> (B,N,Cout)

    mesh_mode:
      - "auto": choose by input rank
      - "grid": force grid
      - "points": force irregular points
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        *,
        mesh_mode: MeshMode = "auto",
        # grid params
        width_grid: int = 64,
        levels_grid: int = 3,
        depth_grid: int = 2,
        modes: Optional[Tuple[int, ...]] = None,
        add_coords_grid: bool = True,
        # point params
        width_points: int = 128,
        levels_points: int = 3,
        depth_points: int = 2,
        k_points: int = 16,
        pool_ratio: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.mesh_mode = mesh_mode

        self.grid_net = UNONdGrid(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            width=width_grid,
            levels=levels_grid,
            depth_per_level=depth_grid,
            modes=modes,
            add_coords=add_coords_grid,
        )

        self.points_net = UNONdPoints(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            width=width_points,
            levels=levels_points,
            depth_per_level=depth_points,
            k=k_points,
            pool_ratio=pool_ratio,
        )

    def forward(
        self,
        u: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        # Decide mode
        mode = self.mesh_mode
        if mode == "auto":
            # grid: (B,C,...) rank >= 3 ; points: (B,N,C) rank == 3 with coords provided
            if coords is not None and u.ndim == 3:
                mode = "points"
            else:
                mode = "grid"

        if mode == "grid":
            y = self.grid_net(u)  # (B,Cout,*spatial)
        elif mode == "points":
            if coords is None:
                raise ValueError("coords must be provided for points mode")
            y = self.points_net(u, coords)  # (B,N,Cout)
        else:
            raise ValueError(f"Invalid mesh_mode: {self.mesh_mode}")

        losses = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return OperatorOutput(y=y, losses=losses, extras={"mode": mode})