"""Production-quality 3D field visualization: streamlines, isosurfaces, volume rendering.

Integrates with matplotlib and optionally scikit-image for isosurface extraction.

Key classes / functions
-----------------------
compute_streamlines      -- RK4 streamline integration on a 2D grid
compute_isosurface       -- marching-cubes isosurface from a 3D scalar field
plot_streamlines_2d      -- model-driven 2D streamline plot
plot_isosurface_3d       -- model-driven 3D isosurface figure
plot_volume_slice        -- 2D slice of a 3D field
FlowVisualizer           -- high-level convenience class
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# matplotlib helper (lazy import, Agg backend so it works headless)
# ---------------------------------------------------------------------------

def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        ) from e


# ---------------------------------------------------------------------------
# Helper: evaluate a PINN model on a grid of points
# ---------------------------------------------------------------------------

def _model_on_grid(
    model: Callable,
    coords: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run model on (N, D) numpy coords, return (N, K) numpy array."""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for model inference")
    import torch
    with torch.no_grad():
        x_t = torch.tensor(coords, dtype=torch.float32, device=device)
        out = model(x_t)
        if isinstance(out, torch.Tensor):
            return out.cpu().numpy()
        return np.asarray(out)


# ---------------------------------------------------------------------------
# Feature 17a: compute_streamlines  (RK4 on a 2D grid)
# ---------------------------------------------------------------------------

def compute_streamlines(
    u_field: np.ndarray,
    v_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    seed_points: Optional[np.ndarray] = None,
    n_seeds: int = 20,
    max_length: float = 1.0,
    dt: float = 0.01,
) -> List[np.ndarray]:
    """Compute 2D streamlines via RK4 integration.

    Parameters
    ----------
    u_field, v_field : (ny, nx) arrays of velocity components
                       (meshgrid 'xy' convention, matching np.meshgrid(x, y))
    x_grid, y_grid   : 1-D coordinate arrays of length nx and ny respectively
    seed_points      : (N, 2) array of starting points; if None, auto-placed on
                       a regular grid spanning the domain
    n_seeds          : number of auto-placed seeds when seed_points is None
    max_length       : maximum integration arc-length (domain units)
    dt               : Euler/RK4 step size

    Returns
    -------
    list of (K, 2) float32 arrays, one per streamline (variable length)
    """
    u = np.asarray(u_field, dtype=np.float64)
    v = np.asarray(v_field, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    def _interp_uv(pt: np.ndarray) -> Tuple[float, float]:
        """Bilinear interpolation of (u, v) at point pt=(px, py)."""
        px, py = pt[0], pt[1]
        # clamp
        if px < x_min or px > x_max or py < y_min or py > y_max:
            return 0.0, 0.0
        # find grid cell
        ix = np.searchsorted(x, px) - 1
        iy = np.searchsorted(y, py) - 1
        ix = int(np.clip(ix, 0, len(x) - 2))
        iy = int(np.clip(iy, 0, len(y) - 2))
        tx = (px - x[ix]) / (x[ix + 1] - x[ix] + 1e-15)
        ty = (py - y[iy]) / (y[iy + 1] - y[iy] + 1e-15)
        # bilinear interpolation (u is (ny, nx))
        u_val = (
            (1 - tx) * (1 - ty) * u[iy, ix]
            + tx * (1 - ty) * u[iy, ix + 1]
            + (1 - tx) * ty * u[iy + 1, ix]
            + tx * ty * u[iy + 1, ix + 1]
        )
        v_val = (
            (1 - tx) * (1 - ty) * v[iy, ix]
            + tx * (1 - ty) * v[iy, ix + 1]
            + (1 - tx) * ty * v[iy + 1, ix]
            + tx * ty * v[iy + 1, ix + 1]
        )
        return float(u_val), float(v_val)

    # auto seed placement
    if seed_points is None:
        nx_s = max(1, int(math.sqrt(n_seeds)))
        ny_s = max(1, n_seeds // nx_s)
        xs = np.linspace(x_min + 0.05 * (x_max - x_min), x_max - 0.05 * (x_max - x_min), nx_s)
        ys = np.linspace(y_min + 0.05 * (y_max - y_min), y_max - 0.05 * (y_max - y_min), ny_s)
        gx, gy = np.meshgrid(xs, ys)
        seed_points = np.stack([gx.ravel(), gy.ravel()], axis=1)

    seed_points = np.asarray(seed_points, dtype=np.float64)
    streamlines = []

    max_steps = int(max_length / (abs(dt) + 1e-15)) + 1

    for seed in seed_points:
        pts = [seed.copy()]
        pos = seed.copy()
        arc = 0.0

        for _ in range(max_steps):
            if arc >= max_length:
                break
            # RK4
            k1u, k1v = _interp_uv(pos)
            k2u, k2v = _interp_uv(pos + 0.5 * dt * np.array([k1u, k1v]))
            k3u, k3v = _interp_uv(pos + 0.5 * dt * np.array([k2u, k2v]))
            k4u, k4v = _interp_uv(pos + dt * np.array([k3u, k3v]))
            du = (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
            dv = (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            step_vec = dt * np.array([du, dv])
            step_len = float(np.linalg.norm(step_vec))
            if step_len < 1e-12:
                break
            pos = pos + step_vec
            arc += step_len
            # boundary check
            if pos[0] < x_min or pos[0] > x_max or pos[1] < y_min or pos[1] > y_max:
                break
            pts.append(pos.copy())

        if len(pts) >= 2:
            streamlines.append(np.array(pts, dtype=np.float32))

    return streamlines


# ---------------------------------------------------------------------------
# Feature 17b: compute_isosurface
# ---------------------------------------------------------------------------

def compute_isosurface(
    field_3d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract isosurface from 3D scalar field using marching cubes.

    Parameters
    ----------
    field_3d : (nx, ny, nz) scalar field array
    x, y, z  : 1-D coordinate arrays of lengths nx, ny, nz
    level    : isovalue

    Returns
    -------
    vertices : (M, 3) float32 array of vertex positions
    faces    : (F, 3) int32 array of triangle face indices

    Raises
    ------
    ImportError if scikit-image is not installed.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for isosurface extraction. "
            "Install with: pip install scikit-image"
        )

    field = np.asarray(field_3d, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0

    verts, faces, _normals, _values = marching_cubes(
        field, level=level, spacing=(dx, dy, dz)
    )
    # shift verts so origin matches x[0], y[0], z[0]
    verts[:, 0] += float(x[0])
    verts[:, 1] += float(y[0])
    verts[:, 2] += float(z[0])

    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Feature 17c: plot_streamlines_2d  (model-driven)
# ---------------------------------------------------------------------------

def plot_streamlines_2d(
    model: Callable,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    n_grid: int = 50,
    n_seeds: int = 30,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Streamlines",
    cmap: str = "viridis",
    savepath: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    show: bool = False,
    background: str = "speed",   # 'speed', 'pressure', or 'none'
) -> Any:
    """Plot 2D streamlines of velocity field predicted by a PINN model.

    Parameters
    ----------
    model     : callable (N, 2) -> (N, K); first two outputs are (u, v).
                Optionally (N, 3) for (u, v, p) -- pressure used as background.
    x_range   : (x_min, x_max)
    y_range   : (y_min, y_max)
    n_grid    : grid resolution for field evaluation
    n_seeds   : number of streamline seed points
    figsize   : matplotlib figure size
    title     : figure title
    cmap      : colormap for background scalar field
    savepath  : optional path to save the figure
    device    : torch device string ('cpu' or 'cuda')
    show      : call plt.show() if True
    background: 'speed' (|u|), 'pressure' (3rd output), or 'none'

    Returns
    -------
    matplotlib Figure
    """
    plt = _get_mpl()

    x_lin = np.linspace(x_range[0], x_range[1], n_grid)
    y_lin = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(x_lin, y_lin, indexing="xy")   # (ny, nx) each
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

    out = _model_on_grid(model, pts, device=device)      # (ny*nx, K)
    u_flat = out[:, 0]
    v_flat = out[:, 1]
    u_grid = u_flat.reshape(n_grid, n_grid)   # (ny, nx)
    v_grid = v_flat.reshape(n_grid, n_grid)

    # background scalar
    bg = None
    if background == "speed":
        bg = np.sqrt(u_grid**2 + v_grid**2)
    elif background == "pressure" and out.shape[1] >= 3:
        p_flat = out[:, 2]
        bg = p_flat.reshape(n_grid, n_grid)

    # compute streamlines
    streams = compute_streamlines(
        u_grid, v_grid, x_lin, y_lin,
        n_seeds=n_seeds,
        max_length=float((x_range[1] - x_range[0]) + (y_range[1] - y_range[0])),
        dt=float(min(x_range[1] - x_range[0], y_range[1] - y_range[0])) / (n_grid * 2),
    )

    fig, ax = plt.subplots(figsize=figsize)

    if bg is not None:
        im = ax.pcolormesh(XX, YY, bg, cmap=cmap, shading="auto", alpha=0.75)
        label = "speed |u|" if background == "speed" else "pressure p"
        fig.colorbar(im, ax=ax, label=label)

    for sl in streams:
        ax.plot(sl[:, 0], sl[:, 1], "k-", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature 17d: plot_isosurface_3d
# ---------------------------------------------------------------------------

def plot_isosurface_3d(
    model: Callable,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
    n_grid: int = 30,
    level: float = 0.0,
    field_idx: int = 0,
    title: str = "Isosurface",
    savepath: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    show: bool = False,
    color: str = "steelblue",
    alpha: float = 0.7,
) -> Any:
    """Plot 3D isosurface of a scalar field predicted by a PINN model.

    Parameters
    ----------
    model     : callable (N, 3) -> (N, K)
    x_range, y_range, z_range : domain extents
    n_grid    : grid resolution per axis
    level     : isovalue
    field_idx : which model output to use (default 0)
    title     : figure title
    savepath  : optional save path
    device    : torch device string
    show      : call plt.show() if True
    color     : face colour of the isosurface mesh
    alpha     : transparency

    Returns
    -------
    matplotlib Figure (with a 3D subplot)

    Requires scikit-image (marching cubes).
    """
    plt = _get_mpl()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x_lin = np.linspace(x_range[0], x_range[1], n_grid)
    y_lin = np.linspace(y_range[0], y_range[1], n_grid)
    z_lin = np.linspace(z_range[0], z_range[1], n_grid)
    XX, YY, ZZ = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")  # (nx,ny,nz)
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)

    out = _model_on_grid(model, pts, device=device)
    field = out[:, field_idx].reshape(n_grid, n_grid, n_grid)

    verts, faces = compute_isosurface(field, x_lin, y_lin, z_lin, level=level)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=alpha, linewidth=0.0)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature 17e: plot_volume_slice
# ---------------------------------------------------------------------------

def plot_volume_slice(
    model: Callable,
    axis: str = "z",
    slice_val: float = 0.5,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    z_range: Tuple[float, float] = (0.0, 1.0),
    n_grid: int = 50,
    field_idx: int = 0,
    cmap: str = "RdBu_r",
    title: str = "Volume slice",
    savepath: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    show: bool = False,
) -> Any:
    """Plot a 2D slice of a 3D field predicted by a PINN model.

    Parameters
    ----------
    model     : callable (N, 3) -> (N, K)
    axis      : 'x', 'y', or 'z' -- axis along which to slice
    slice_val : coordinate value on the slicing axis
    x_range, y_range, z_range : domain extents
    n_grid    : grid resolution on each free axis
    field_idx : which model output to plot
    cmap      : matplotlib colormap
    title     : figure title
    savepath  : optional save path
    device    : torch device string
    show      : call plt.show() if True

    Returns
    -------
    matplotlib Figure
    """
    plt = _get_mpl()

    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    x_lin = np.linspace(x_range[0], x_range[1], n_grid)
    y_lin = np.linspace(y_range[0], y_range[1], n_grid)
    z_lin = np.linspace(z_range[0], z_range[1], n_grid)

    if axis == "z":
        A_lin, B_lin = x_lin, y_lin
        AA, BB = np.meshgrid(A_lin, B_lin, indexing="ij")
        CC = np.full_like(AA, slice_val)
        pts = np.stack([AA.ravel(), BB.ravel(), CC.ravel()], axis=1)
        xlabel, ylabel = "x", "y"
    elif axis == "y":
        A_lin, B_lin = x_lin, z_lin
        AA, BB = np.meshgrid(A_lin, B_lin, indexing="ij")
        CC = np.full_like(AA, slice_val)
        pts = np.stack([AA.ravel(), CC.ravel(), BB.ravel()], axis=1)
        xlabel, ylabel = "x", "z"
    else:  # axis == "x"
        A_lin, B_lin = y_lin, z_lin
        AA, BB = np.meshgrid(A_lin, B_lin, indexing="ij")
        CC = np.full_like(AA, slice_val)
        pts = np.stack([CC.ravel(), AA.ravel(), BB.ravel()], axis=1)
        xlabel, ylabel = "y", "z"

    pts = pts.astype(np.float32)
    out = _model_on_grid(model, pts, device=device)
    field_2d = out[:, field_idx].reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(AA, BB, field_2d, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=f"field[{field_idx}]")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Slice at {axis}={slice_val:.3g}")
    ax.set_aspect("equal")

    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Feature 17f: FlowVisualizer  (high-level class)
# ---------------------------------------------------------------------------

class FlowVisualizer:
    """High-level flow visualization class.

    Wraps a PINN model and provides one-liner plotting methods.

    Parameters
    ----------
    model        : PINN model callable (N, D) -> (N, K)
    coord_ranges : dict mapping axis name to (min, max) e.g.
                   {'x': (0,4), 'y': (0,1)} for 2D or
                   {'x': (0,1), 'y': (0,1), 'z': (0,1)} for 3D
    device       : torch device string ('cpu' or 'cuda')

    Example::

        vis = FlowVisualizer(pinn_model, coord_ranges={'x': (0,4), 'y': (0,1)})
        vis.streamlines(n_seeds=40, savepath="streamlines.png")
        vis.pressure_contours(savepath="pressure.png")
        vis.vorticity(savepath="vorticity.png")
    """

    def __init__(
        self,
        model: Callable,
        coord_ranges: Dict[str, Tuple[float, float]],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.coord_ranges = coord_ranges
        self.device = device
        self._axes = list(coord_ranges.keys())
        self._dim = len(self._axes)
        if self._dim not in (2, 3):
            raise ValueError(f"coord_ranges must have 2 or 3 keys, got {self._dim}")

    def _x_range(self) -> Tuple[float, float]:
        return self.coord_ranges[self._axes[0]]

    def _y_range(self) -> Tuple[float, float]:
        return self.coord_ranges[self._axes[1]]

    def _z_range(self) -> Tuple[float, float]:
        if self._dim < 3:
            raise ValueError("3D operation requires 3 coord axes")
        return self.coord_ranges[self._axes[2]]

    def _eval_2d_grid(self, n_grid: int, field_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on 2D grid. Returns (XX, YY, field_2d)."""
        x_lin = np.linspace(*self._x_range(), n_grid)
        y_lin = np.linspace(*self._y_range(), n_grid)
        XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
        out = _model_on_grid(self.model, pts, device=self.device)
        field = out[:, field_idx].reshape(n_grid, n_grid)
        return XX, YY, field

    def _eval_2d_uv(self, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (x_lin, y_lin, u_grid, v_grid, out_all) on 2D grid."""
        x_lin = np.linspace(*self._x_range(), n_grid)
        y_lin = np.linspace(*self._y_range(), n_grid)
        XX, YY = np.meshgrid(x_lin, y_lin, indexing="xy")   # (ny, nx)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
        out = _model_on_grid(self.model, pts, device=self.device)
        u = out[:, 0].reshape(n_grid, n_grid)
        v = out[:, 1].reshape(n_grid, n_grid)
        return x_lin, y_lin, u, v, out

    def streamlines(
        self,
        n_grid: int = 60,
        n_seeds: int = 30,
        figsize: Tuple[float, float] = (10, 6),
        title: str = "Streamlines",
        cmap: str = "viridis",
        savepath: Optional[Union[str, Path]] = None,
        show: bool = False,
        **kwargs,
    ) -> Any:
        """2D streamline plot. Model must output (u, v, ...) as first two fields."""
        if self._dim != 2:
            raise ValueError("streamlines() requires a 2D domain (2 coord axes)")
        return plot_streamlines_2d(
            self.model,
            x_range=self._x_range(),
            y_range=self._y_range(),
            n_grid=n_grid,
            n_seeds=n_seeds,
            figsize=figsize,
            title=title,
            cmap=cmap,
            savepath=savepath,
            device=self.device,
            show=show,
            **kwargs,
        )

    def pressure_contours(
        self,
        n_grid: int = 60,
        n_levels: int = 20,
        cmap: str = "RdBu_r",
        title: str = "Pressure field",
        savepath: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """2D pressure contour plot. Model must output (..., p) at index 2."""
        plt = _get_mpl()

        if self._dim != 2:
            raise ValueError("pressure_contours() requires a 2D domain")

        n_grid_eff = n_grid
        x_lin = np.linspace(*self._x_range(), n_grid_eff)
        y_lin = np.linspace(*self._y_range(), n_grid_eff)
        XX, YY = np.meshgrid(x_lin, y_lin, indexing="ij")
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
        out = _model_on_grid(self.model, pts, device=self.device)
        p_idx = 2 if out.shape[1] >= 3 else 0
        p_grid = out[:, p_idx].reshape(n_grid_eff, n_grid_eff)

        fig, ax = plt.subplots(figsize=(8, 5))
        cf = ax.contourf(XX, YY, p_grid, levels=n_levels, cmap=cmap)
        ax.contour(XX, YY, p_grid, levels=n_levels, colors="k", linewidths=0.4, alpha=0.5)
        fig.colorbar(cf, ax=ax, label="pressure p")
        ax.set_xlabel(self._axes[0])
        ax.set_ylabel(self._axes[1])
        ax.set_title(title)
        ax.set_aspect("equal")

        if savepath is not None:
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        if not show:
            plt.close(fig)
        return fig

    def velocity_magnitude(
        self,
        n_grid: int = 60,
        cmap: str = "plasma",
        title: str = "Velocity magnitude |u|",
        savepath: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """2D velocity magnitude plot."""
        plt = _get_mpl()

        if self._dim != 2:
            raise ValueError("velocity_magnitude() requires a 2D domain")

        x_lin, y_lin, u, v, _ = self._eval_2d_uv(n_grid)
        speed = np.sqrt(u**2 + v**2)
        XX, YY = np.meshgrid(x_lin, y_lin, indexing="xy")

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.pcolormesh(XX, YY, speed, cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax, label="|u|")
        ax.set_xlabel(self._axes[0])
        ax.set_ylabel(self._axes[1])
        ax.set_title(title)
        ax.set_aspect("equal")

        if savepath is not None:
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        if not show:
            plt.close(fig)
        return fig

    def vorticity(
        self,
        n_grid: int = 60,
        cmap: str = "RdBu_r",
        title: str = "Vorticity dv/dx - du/dy",
        savepath: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> Any:
        """Compute and plot vorticity omega_z = dv/dx - du/dy via autograd.

        Requires torch to be installed.  Falls back to finite differences if
        autograd is unavailable or model does not support it.
        """
        plt = _get_mpl()

        if self._dim != 2:
            raise ValueError("vorticity() requires a 2D domain")

        x_lin = np.linspace(*self._x_range(), n_grid)
        y_lin = np.linspace(*self._y_range(), n_grid)
        XX, YY = np.meshgrid(x_lin, y_lin, indexing="xy")   # (ny, nx)
        pts_np = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

        vort_grid = None
        if _TORCH_AVAILABLE:
            import torch
            try:
                pts_t = torch.tensor(pts_np, dtype=torch.float32,
                                     device=self.device, requires_grad=True)
                out = self.model(pts_t)
                u_t = out[:, 0:1]
                v_t = out[:, 1:2]
                ones = torch.ones_like(u_t)
                grads_u = torch.autograd.grad(u_t, pts_t, ones, create_graph=False)[0]
                grads_v = torch.autograd.grad(v_t, pts_t, ones, create_graph=False)[0]
                du_dy = grads_u[:, 1].detach().cpu().numpy()
                dv_dx = grads_v[:, 0].detach().cpu().numpy()
                vort_grid = (dv_dx - du_dy).reshape(n_grid, n_grid)
            except Exception:
                vort_grid = None

        if vort_grid is None:
            # finite-difference fallback
            out = _model_on_grid(self.model, pts_np, device=self.device)
            u_g = out[:, 0].reshape(n_grid, n_grid)
            v_g = out[:, 1].reshape(n_grid, n_grid)
            du_dy = np.gradient(u_g, y_lin, axis=1)
            dv_dx = np.gradient(v_g, x_lin, axis=0)
            vort_grid = dv_dx - du_dy

        vmax = float(np.max(np.abs(vort_grid))) or 1.0

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.pcolormesh(XX, YY, vort_grid, cmap=cmap, shading="auto",
                           vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label="omega_z")
        ax.set_xlabel(self._axes[0])
        ax.set_ylabel(self._axes[1])
        ax.set_title(title)
        ax.set_aspect("equal")

        if savepath is not None:
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        if not show:
            plt.close(fig)
        return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "compute_streamlines",
    "compute_isosurface",
    "plot_streamlines_2d",
    "plot_isosurface_3d",
    "plot_volume_slice",
    "FlowVisualizer",
]
