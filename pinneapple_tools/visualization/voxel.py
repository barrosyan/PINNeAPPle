"""Visualisation of VoxelGrid objects."""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import get_cmap, make_figure, DEFAULT_CMAP


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


def plot_voxel_slice(
    grid,   # VoxelGrid
    *,
    axis: str = "z",
    index: Optional[int] = None,
    cmap: str = DEFAULT_CMAP,
    title: str = "",
    show: bool = False,
) -> Figure:
    """
    Plot a single 2-D slice through a 3-D VoxelGrid.

    axis  : "x" | "y" | "z" — slice plane normal
    index : voxel index along that axis (defaults to midpoint)
    """
    data = _to_np(grid.data)
    if data.ndim == 2:
        # Already 2-D
        fig, ax = make_figure(figsize=(7, 6))
        im = ax.imshow(data.T, origin="lower", cmap=get_cmap(cmap), aspect="equal")
        fig.colorbar(im, ax=ax, label="value")
        ax.set_title(title or "VoxelGrid 2-D")
        ax.set_xlabel("voxel i"); ax.set_ylabel("voxel j")
        if show:
            plt.show()
        return fig

    ax_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = ax_map[axis]
    mid = data.shape[ax_idx] // 2 if index is None else int(index)
    slice_data = np.take(data, mid, axis=ax_idx)   # (ny,nz) etc.

    remaining = [a for a in ["x", "y", "z"] if a != axis]
    fig, ax = make_figure(figsize=(7, 6))
    im = ax.imshow(slice_data.T, origin="lower", cmap=get_cmap(cmap), aspect="equal")
    fig.colorbar(im, ax=ax, label="value")
    ax.set_xlabel(remaining[0]); ax.set_ylabel(remaining[1])
    ax.set_title(title or f"VoxelGrid slice  {axis}={mid}")
    if show:
        plt.show()
    return fig


def plot_voxel_3d(
    grid,   # VoxelGrid
    *,
    threshold: float = 0.5,
    cmap: str = DEFAULT_CMAP,
    alpha: float = 0.6,
    title: str = "Voxel occupancy",
    show: bool = False,
    max_voxels: int = 50_000,
) -> Figure:
    """
    Scatter-based 3-D visualisation of occupied voxels.

    threshold : voxels with data > threshold are shown.
    max_voxels: subsample if there are more than this many occupied voxels.
    """
    data = _to_np(grid.data)
    if data.ndim == 2:
        return plot_voxel_slice(grid, title=title, cmap=cmap, show=show)

    ijk = np.argwhere(data > threshold)
    values = data[data > threshold]

    if ijk.shape[0] > max_voxels:
        idx = np.random.choice(ijk.shape[0], max_voxels, replace=False)
        ijk = ijk[idx]
        values = values[idx]

    # Convert to world coords
    origin = _to_np(grid.origin)
    vs = _to_np(grid.voxel_size) if hasattr(grid.voxel_size, "numpy") else np.full(3, grid.voxel_size)
    xyz = origin + (ijk + 0.5) * vs

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    c=values, cmap=get_cmap(cmap), s=3, alpha=alpha)
    fig.colorbar(sc, ax=ax, shrink=0.5, label="value")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"{title}  |  {ijk.shape[0]} voxels")
    if show:
        plt.show()
    return fig


def plot_voxel_histogram(
    grid,   # VoxelGrid
    *,
    bins: int = 50,
    title: str = "Voxel value distribution",
    show: bool = False,
) -> Figure:
    """Histogram of non-zero voxel values."""
    data = _to_np(grid.data).ravel()
    occupied = data[data > 0]

    fig, ax = make_figure(figsize=(6, 4))
    ax.hist(occupied, bins=bins, color="#4c9be8", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if show:
        plt.show()
    return fig
