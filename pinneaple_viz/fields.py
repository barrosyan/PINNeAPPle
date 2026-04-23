"""
2-D / 3-D scalar and vector field plots — CFD style.

Typical usage
-------------
from pinneaple_viz.fields import plot_scalar, plot_vectors, plot_streamlines

plot_scalar(xc, yc, pressure, title="Pressure [Pa]", cmap="coolwarm")
plot_vectors(xc, yc, ux, uy, title="Velocity field")
plot_streamlines(xc, yc, ux, uy)
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .style import get_cmap, make_figure, DEFAULT_CMAP


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


# ---------------------------------------------------------------------------
# Scalar field
# ---------------------------------------------------------------------------

def plot_scalar(
    x: "ArrayLike",
    y: "ArrayLike",
    field: "ArrayLike",
    *,
    title: str = "Scalar field",
    label: str = "",
    cmap: str = DEFAULT_CMAP,
    n_levels: int = 64,
    show_contour_lines: bool = True,
    n_lines: int = 12,
    axes: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False,
) -> Figure:
    """
    Filled contour plot of a 2-D scalar field.

    x, y    : 1-D or 2-D arrays of coordinates
    field   : matching array of scalar values
    """
    x_ = _to_np(x).ravel()
    y_ = _to_np(y).ravel()
    f_ = _to_np(field).ravel()

    if axes is None:
        fig, ax = make_figure(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    cm = get_cmap(cmap)
    vmin_ = vmin if vmin is not None else f_.min()
    vmax_ = vmax if vmax is not None else f_.max()

    # Structured grid path (faster)
    if x_.shape == y_.shape == f_.shape:
        nx = len(np.unique(np.round(x_, 10)))
        ny = len(np.unique(np.round(y_, 10)))
        if nx * ny == len(x_):
            X = x_.reshape(nx, ny)
            Y = y_.reshape(nx, ny)
            F = f_.reshape(nx, ny)
            cf = ax.contourf(X, Y, F, levels=n_levels, cmap=cm, vmin=vmin_, vmax=vmax_)
            if show_contour_lines:
                ax.contour(X, Y, F, levels=n_lines, colors="white", linewidths=0.4, alpha=0.5)
        else:
            triang = mtri.Triangulation(x_, y_)
            cf = ax.tricontourf(triang, f_, levels=n_levels, cmap=cm, vmin=vmin_, vmax=vmax_)
            if show_contour_lines:
                ax.tricontour(triang, f_, levels=n_lines, colors="white", linewidths=0.4, alpha=0.5)
    else:
        triang = mtri.Triangulation(x_, y_)
        cf = ax.tricontourf(triang, f_, levels=n_levels, cmap=cm, vmin=vmin_, vmax=vmax_)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(label or title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    if show:
        plt.show()
    return fig


def plot_scalar_3d(
    x: "ArrayLike",
    y: "ArrayLike",
    z: "ArrayLike",
    field: "ArrayLike",
    *,
    slice_axis: str = "z",
    slice_val: Optional[float] = None,
    title: str = "Scalar field (3-D slice)",
    cmap: str = DEFAULT_CMAP,
    show: bool = False,
) -> Figure:
    """Plot a 2-D slice of a 3-D scalar field."""
    x_ = _to_np(x).ravel()
    y_ = _to_np(y).ravel()
    z_ = _to_np(z).ravel()
    f_ = _to_np(field).ravel()

    ax_map = {"x": x_, "y": y_, "z": z_}
    slice_coords = ax_map[slice_axis]
    sv = slice_val if slice_val is not None else np.median(slice_coords)
    tol = (slice_coords.max() - slice_coords.min()) / 20.0
    mask = np.abs(slice_coords - sv) < tol

    remaining = {"x": x_, "y": y_, "z": z_}
    remaining.pop(slice_axis)
    (a_name, a_vals), (b_name, b_vals) = list(remaining.items())

    fig, ax = make_figure()
    triang = mtri.Triangulation(a_vals[mask], b_vals[mask])
    cf = ax.tricontourf(triang, f_[mask], levels=64, cmap=get_cmap(cmap))
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel(a_name); ax.set_ylabel(b_name)
    ax.set_title(f"{title}  |  {slice_axis}={sv:.3g}")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

def plot_vectors(
    x: "ArrayLike",
    y: "ArrayLike",
    u: "ArrayLike",
    v: "ArrayLike",
    *,
    title: str = "Vector field",
    scale: Optional[float] = None,
    density: int = 20,
    color_by_mag: bool = True,
    cmap: str = "rainbow",
    background_field: Optional["ArrayLike"] = None,
    bg_cmap: str = "coolwarm",
    bg_label: str = "",
    axes: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    show: bool = False,
) -> Figure:
    """
    Quiver plot with optional background scalar field.

    density : approximate number of arrows along each axis
    """
    x_ = _to_np(x).ravel()
    y_ = _to_np(y).ravel()
    u_ = _to_np(u).ravel()
    v_ = _to_np(v).ravel()

    if axes is None:
        fig, ax = make_figure(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    if background_field is not None:
        bg_ = _to_np(background_field).ravel()
        triang = mtri.Triangulation(x_, y_)
        cf = ax.tricontourf(triang, bg_, levels=64, cmap=get_cmap(bg_cmap), alpha=0.85)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label(bg_label, fontsize=9)

    # Subsample
    n = len(x_)
    step = max(1, n // (density * density))
    idx = np.arange(0, n, step)

    mag = np.sqrt(u_[idx] ** 2 + v_[idx] ** 2)
    cm = get_cmap(cmap) if color_by_mag else None
    qv = ax.quiver(
        x_[idx], y_[idx], u_[idx], v_[idx],
        mag if color_by_mag else None,
        cmap=cm,
        scale=scale,
        alpha=0.9,
    )
    if color_by_mag:
        fig.colorbar(qv, ax=ax, label="|v|")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Streamlines
# ---------------------------------------------------------------------------

def plot_streamlines(
    x: "ArrayLike",
    y: "ArrayLike",
    u: "ArrayLike",
    v: "ArrayLike",
    *,
    title: str = "Streamlines",
    density: float = 1.5,
    color_by_mag: bool = True,
    cmap: str = "plasma",
    linewidth_scale: float = 1.5,
    background_field: Optional["ArrayLike"] = None,
    bg_cmap: str = "coolwarm",
    axes: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    show: bool = False,
) -> Figure:
    """
    Streamline plot (requires structured or interpolated regular grid).
    Handles scattered points by interpolating onto a regular grid.
    """
    import scipy.interpolate as si

    x_ = _to_np(x).ravel()
    y_ = _to_np(y).ravel()
    u_ = _to_np(u).ravel()
    v_ = _to_np(v).ravel()

    if axes is None:
        fig, ax = make_figure(figsize=figsize)
    else:
        ax = axes
        fig = ax.get_figure()

    # Interpolate onto regular grid
    nx_g = ny_g = 80
    xi = np.linspace(x_.min(), x_.max(), nx_g)
    yi = np.linspace(y_.min(), y_.max(), ny_g)
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")
    pts = np.column_stack([x_, y_])
    Ui = si.griddata(pts, u_, (Xi, Yi), method="linear", fill_value=0.0)
    Vi = si.griddata(pts, v_, (Xi, Yi), method="linear", fill_value=0.0)
    mag = np.sqrt(Ui ** 2 + Vi ** 2)

    if background_field is not None:
        bg_ = _to_np(background_field).ravel()
        BG = si.griddata(pts, bg_, (Xi, Yi), method="linear")
        cf = ax.contourf(Xi, Yi, BG, levels=64, cmap=get_cmap(bg_cmap), alpha=0.7)
        fig.colorbar(cf, ax=ax, pad=0.02)

    lw = linewidth_scale * mag / (mag.max() + 1e-10) + 0.4
    strm = ax.streamplot(
        xi, yi, Ui.T, Vi.T,
        color=mag.T if color_by_mag else "white",
        cmap=get_cmap(cmap) if color_by_mag else None,
        linewidth=lw.T,
        density=density,
        arrowsize=1.2,
    )
    if color_by_mag:
        fig.colorbar(strm.lines, ax=ax, label="|v|")

    ax.set_xlim(x_.min(), x_.max())
    ax.set_ylim(y_.min(), y_.max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-panel field comparison
# ---------------------------------------------------------------------------

def compare_fields(
    x: "ArrayLike",
    y: "ArrayLike",
    fields: Sequence[Tuple[str, "ArrayLike"]],
    *,
    cmap: str = DEFAULT_CMAP,
    n_cols: int = 3,
    figsize_per_panel: Tuple[int, int] = (5, 4),
    suptitle: str = "",
    show: bool = False,
) -> Figure:
    """
    Side-by-side contour comparison of several fields.

    fields : list of (label, array) pairs
    """
    n = len(fields)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    fw = figsize_per_panel[0] * n_cols
    fh = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh))
    axes = np.array(axes).ravel()

    for i, (label, field) in enumerate(fields):
        plot_scalar(x, y, field, title=label, cmap=cmap, axes=axes[i], label=label)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Residual / error map
# ---------------------------------------------------------------------------

def plot_error(
    x: "ArrayLike",
    y: "ArrayLike",
    pred: "ArrayLike",
    ref: "ArrayLike",
    *,
    relative: bool = True,
    title: str = "Prediction error",
    cmap: str = "hot",
    show: bool = False,
) -> Figure:
    """Visualise pointwise |pred - ref| (or relative error) on the domain."""
    p_ = _to_np(pred).ravel()
    r_ = _to_np(ref).ravel()
    err = np.abs(p_ - r_)
    if relative:
        scale = np.abs(r_).max()
        err = err / (scale + 1e-12)
        label = "Relative error"
    else:
        label = "Absolute error"

    fig = plot_scalar(x, y, err, title=title, label=label, cmap=cmap, show=show)
    return fig
