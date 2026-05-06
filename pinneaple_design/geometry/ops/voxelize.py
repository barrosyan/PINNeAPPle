"""Voxelization and meshfree sampling utilities for PINNeAPPle geometries."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch


# ---------------------------------------------------------------------------
# VoxelGrid dataclass
# ---------------------------------------------------------------------------

@dataclass
class VoxelGrid:
    """
    Regular axis-aligned 3-D (or 2-D) voxel grid.

    Attributes
    ----------
    data      : float tensor of shape (Cx, Cy[, Cz]) — one value per voxel
    origin    : (3,) or (2,) lower-left corner in world space
    voxel_size: scalar edge length (isotropic) or (dx, dy[, dz]) tuple
    dim       : 2 or 3
    meta      : arbitrary extras (field name, units, …)
    """
    data: torch.Tensor
    origin: torch.Tensor
    voxel_size: Union[float, torch.Tensor]
    dim: int = 3
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def resolution(self) -> Tuple[int, ...]:
        return self.shape

    def _step(self) -> torch.Tensor:
        vs = self.voxel_size
        if isinstance(vs, (int, float)):
            return torch.full((self.dim,), float(vs), device=self.origin.device)
        return vs.to(self.origin.device)

    def to_world(self, ijk: torch.Tensor) -> torch.Tensor:
        """Convert integer voxel indices (…, dim) → world coords."""
        return self.origin + (ijk.float() + 0.5) * self._step()

    def to_voxel(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert world coords (…, dim) → integer voxel indices (floor)."""
        return ((xyz - self.origin) / self._step()).floor().long()

    def occupied_centres(self, threshold: float = 0.5) -> torch.Tensor:
        """Return (N, dim) world coordinates of voxels where data > threshold."""
        idx = torch.nonzero(self.data > threshold, as_tuple=False)
        return self.to_world(idx)

    def to_collocation_points(self, threshold: float = 0.5) -> torch.Tensor:
        """Alias for occupied_centres — used in PINNeAPPle pipeline."""
        return self.occupied_centres(threshold)

    def __repr__(self) -> str:  # noqa: D105
        vs = self.voxel_size
        return f"VoxelGrid(shape={self.shape}, voxel_size={vs}, dim={self.dim})"


# ---------------------------------------------------------------------------
# Core voxelizers
# ---------------------------------------------------------------------------

def voxelize_domain(
    bounds: Dict[str, Tuple[float, float]],
    resolution: Union[int, Tuple[int, ...]],
    device: torch.device = torch.device("cpu"),
) -> VoxelGrid:
    """
    Create a fully-occupied VoxelGrid for a rectangular domain.

    Parameters
    ----------
    bounds      : {"x": (x0,x1), "y": (y0,y1)[, "z": (z0,z1)]}
    resolution  : int (isotropic) or (nx, ny[, nz])
    """
    axes = [k for k in ("x", "y", "z") if k in bounds]
    dim = len(axes)
    lo = torch.tensor([bounds[a][0] for a in axes], dtype=torch.float32, device=device)
    hi = torch.tensor([bounds[a][1] for a in axes], dtype=torch.float32, device=device)

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(int(r) for r in resolution)

    vs = (hi - lo) / torch.tensor(res, dtype=torch.float32, device=device)
    data = torch.ones(*res, device=device)
    return VoxelGrid(data=data, origin=lo, voxel_size=vs, dim=dim)


def voxelize_sdf(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: Dict[str, Tuple[float, float]],
    resolution: Union[int, Tuple[int, ...]],
    *,
    threshold: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> VoxelGrid:
    """
    Voxelize a signed-distance field: voxel is occupied when sdf ≤ threshold.

    Parameters
    ----------
    sdf_fn      : callable (N, dim) → (N,) SDF values
    bounds      : axis bounds dict
    resolution  : int or tuple
    threshold   : voxel inside if sdf ≤ threshold (0 = on-surface)
    """
    axes = [k for k in ("x", "y", "z") if k in bounds]
    dim = len(axes)
    lo = torch.tensor([bounds[a][0] for a in axes], dtype=torch.float32, device=device)
    hi = torch.tensor([bounds[a][1] for a in axes], dtype=torch.float32, device=device)

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(int(r) for r in resolution)

    # Build dense grid of query points
    grids = [torch.linspace(lo[i].item() + (hi[i] - lo[i]).item() / (2 * res[i]),
                            hi[i].item()  - (hi[i] - lo[i]).item() / (2 * res[i]),
                            res[i], device=device) for i in range(dim)]

    mg = torch.meshgrid(*grids, indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in mg], dim=1)  # (N, dim)
    sdf_vals = sdf_fn(pts)                                   # (N,)

    occupied = (sdf_vals <= threshold).reshape(*res).float()
    vs = (hi - lo) / torch.tensor(res, dtype=torch.float32, device=device)
    return VoxelGrid(data=occupied, origin=lo, voxel_size=vs, dim=dim)


def voxelize_pointcloud(
    points: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]],
    resolution: Union[int, Tuple[int, ...]],
    *,
    values: Optional[torch.Tensor] = None,
    aggregate: str = "mean",
    device: torch.device = torch.device("cpu"),
) -> VoxelGrid:
    """
    Rasterize a point cloud into a voxel grid.

    Parameters
    ----------
    points    : (N, dim) world coordinates
    bounds    : axis bounds dict
    resolution: int or tuple
    values    : (N,) scalar field to aggregate; if None, grid stores occupancy count
    aggregate : "mean" | "sum" | "max" | "min" | "count"
    """
    points = points.to(device)
    axes = [k for k in ("x", "y", "z") if k in bounds]
    dim = len(axes)
    lo = torch.tensor([bounds[a][0] for a in axes], dtype=torch.float32, device=device)
    hi = torch.tensor([bounds[a][1] for a in axes], dtype=torch.float32, device=device)

    if isinstance(resolution, int):
        res = (resolution,) * dim
    else:
        res = tuple(int(r) for r in resolution)

    vs_t = (hi - lo) / torch.tensor(res, dtype=torch.float32, device=device)
    data = torch.zeros(*res, device=device)
    count = torch.zeros(*res, device=device)

    if values is None:
        values_t = torch.ones(points.shape[0], device=device)
        aggregate = "sum"
    else:
        values_t = values.to(device).float()

    ijk = ((points[:, :dim] - lo) / vs_t).floor().long()
    valid = ((ijk >= 0) & (ijk < torch.tensor(res, device=device))).all(dim=1)
    ijk = ijk[valid]
    v   = values_t[valid]

    for n in range(ijk.shape[0]):
        idx = tuple(ijk[n, d].item() for d in range(dim))
        data[idx] += v[n]
        count[idx] += 1.0

    if aggregate == "mean":
        mask = count > 0
        data[mask] = data[mask] / count[mask]
    elif aggregate in ("sum", "count"):
        pass  # already accumulated
    elif aggregate == "max":
        # redo with scatter — mean path above is good enough for count-weighted
        data = torch.zeros(*res, device=device)
        for n in range(ijk.shape[0]):
            idx = tuple(ijk[n, d].item() for d in range(dim))
            data[idx] = max(data[idx].item(), v[n].item())
    elif aggregate == "min":
        data = torch.full(res, float("inf"), device=device)
        for n in range(ijk.shape[0]):
            idx = tuple(ijk[n, d].item() for d in range(dim))
            data[idx] = min(data[idx].item(), v[n].item())
        data[data == float("inf")] = 0.0

    return VoxelGrid(data=data, origin=lo, voxel_size=vs_t, dim=dim)


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------

def voxelgrid_to_collocation(
    grid: VoxelGrid,
    threshold: float = 0.5,
    include_values: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract occupied voxel centres as collocation points for PINNeAPPle.

    Returns
    -------
    pts        : (N, dim) tensor of world coordinates
    vals       : (N,) scalar values (only if include_values=True)
    """
    pts = grid.occupied_centres(threshold)
    if not include_values:
        return pts
    ijk = grid.to_voxel(pts)
    vals = torch.stack([grid.data[tuple(ijk[n, d].item() for d in range(grid.dim))]
                        for n in range(ijk.shape[0])])
    return pts, vals


def sample_voxelgrid(
    grid: VoxelGrid,
    n_samples: int,
    strategy: str = "random",
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Sample collocation points from a VoxelGrid.

    strategy:
      "random"  — uniform random from occupied voxels
      "all"     — every occupied voxel centre
      "surface" — voxels adjacent to empty voxels (surface shell)
    """
    pts = grid.occupied_centres(threshold)
    if strategy == "all" or pts.shape[0] <= n_samples:
        return pts

    if strategy == "random":
        idx = torch.randperm(pts.shape[0])[:n_samples]
        return pts[idx]

    if strategy == "surface":
        # Detect surface: any occupied voxel adjacent to an unoccupied one
        occ = (grid.data > threshold).float()
        # Max-pool over 3x3(x3) neighbourhood; surface = occ & (neighbourhood < 1)
        # simple 2-D/3-D dilation via unfold
        if grid.dim == 2:
            padded = torch.nn.functional.pad(occ.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1))
            neigh  = torch.nn.functional.max_pool2d(padded, 3, stride=1, padding=0)[0, 0]
            surface_mask = (occ > 0.5) & (neigh < 1.0 - 1e-6)
        else:
            padded = torch.nn.functional.pad(occ.unsqueeze(0).unsqueeze(0), (1,)*6)
            neigh  = torch.nn.functional.max_pool3d(padded, 3, stride=1, padding=0)[0, 0]
            surface_mask = (occ > 0.5) & (neigh < 1.0 - 1e-6)
        s_idx = torch.nonzero(surface_mask, as_tuple=False)
        s_pts = grid.to_world(s_idx)
        if s_pts.shape[0] <= n_samples:
            return s_pts
        idx = torch.randperm(s_pts.shape[0])[:n_samples]
        return s_pts[idx]

    raise ValueError(f"Unknown strategy: {strategy!r}")
