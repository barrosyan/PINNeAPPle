"""Visualise SolverOutput objects from FDM / FEM / FVM / meshfree solvers."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import get_cmap, make_figure, DEFAULT_CMAP
from .fields import plot_scalar, plot_vectors, plot_streamlines, compare_fields


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


# ---------------------------------------------------------------------------
# Generic SolverOutput viewer
# ---------------------------------------------------------------------------

def plot_solver_output(
    output,  # SolverOutput
    *,
    x: Optional["ArrayLike"] = None,
    y: Optional["ArrayLike"] = None,
    title: str = "Solver result",
    cmap: str = DEFAULT_CMAP,
    show: bool = False,
) -> Figure:
    """
    Auto-plot a SolverOutput.

    If result is 2-D → contour map.
    If result is 3-D (trajectory: nt, nx, ny) → shows first, middle and last frame.
    If result is 1-D → line plot.
    """
    res = _to_np(output.result)

    if res.ndim == 1:
        fig, ax = make_figure()
        if x is not None:
            ax.plot(_to_np(x), res, lw=1.5)
            ax.set_xlabel("x")
        else:
            ax.plot(res, lw=1.5)
        ax.set_title(title)
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)

    elif res.ndim == 2:
        # Structured (nx, ny) grid
        nx, ny = res.shape
        if x is None:
            x_ = np.linspace(0, 1, nx)
            y_ = np.linspace(0, 1, ny)
        else:
            x_ = _to_np(x)
            y_ = _to_np(y)
        if x_.ndim == 1 and y_.ndim == 1:
            X, Y = np.meshgrid(x_, y_, indexing="ij")
        else:
            X, Y = x_, y_
        fig = plot_scalar(X.ravel(), Y.ravel(), res.ravel(), title=title, cmap=cmap)

    elif res.ndim == 3:
        # Trajectory: (nt+1, nx, ny)
        nt, nx, ny = res.shape
        frames = [0, nt // 2, nt - 1]
        if x is None:
            x_ = np.linspace(0, 1, nx)
            y_ = np.linspace(0, 1, ny)
        else:
            x_ = _to_np(x)
            y_ = _to_np(y)
        X, Y = np.meshgrid(x_, y_, indexing="ij")

        panels = [(f"t-step {k}", res[k]) for k in frames]
        vmin = res.min(); vmax = res.max()
        fig = compare_fields(
            X.ravel(), Y.ravel(), panels, cmap=cmap, n_cols=3,
            suptitle=title,
        )
    else:
        raise ValueError(f"Cannot auto-plot result with ndim={res.ndim}")

    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# FEM mesh + solution
# ---------------------------------------------------------------------------

def plot_fem_result(
    output,  # SolverOutput from FEMSolver
    *,
    component: int = 0,
    title: str = "FEM solution",
    cmap: str = DEFAULT_CMAP,
    show_mesh: bool = True,
    show: bool = False,
) -> Figure:
    """
    Plot FEM solution on its Q1 mesh.

    output.extras must contain "nodes" (N,2) and optionally "elems" (E,4).
    For elasticity results (2N DOFs) component selects ur (0) or uz (1).
    """
    extras = output.extras
    nodes = _to_np(extras["nodes"])   # (N, 2)
    res   = _to_np(output.result)

    if res.ravel().shape[0] == 2 * nodes.shape[0]:
        # Elasticity: interleaved [ur0,uz0,...]
        res = res.ravel()[component::2]
    else:
        res = res.ravel()

    fig, ax = make_figure(figsize=(8, 6))
    triang_val = plt.matplotlib.tri.Triangulation(nodes[:, 0], nodes[:, 1])
    cf = ax.tricontourf(triang_val, res, levels=64, cmap=get_cmap(cmap))
    fig.colorbar(cf, ax=ax, pad=0.02, label=["ur", "uz"][component] if len(res) * 2 == res.size else "u")

    if show_mesh and "elems" in extras:
        elems = _to_np(extras["elems"])
        for e in elems[::max(1, len(elems) // 500)]:
            xs = nodes[np.append(e, e[0]), 0]
            ys = nodes[np.append(e, e[0]), 1]
            ax.plot(xs, ys, "w-", lw=0.2, alpha=0.3)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# FVM cell solution
# ---------------------------------------------------------------------------

def plot_fvm_result(
    output,  # SolverOutput from FVMSolver
    *,
    frame: int = -1,
    title: str = "FVM solution",
    cmap: str = DEFAULT_CMAP,
    show: bool = False,
) -> Figure:
    """
    Plot a single time-step from an FVM trajectory.

    output.result is (nt+1, nx, ny).
    output.extras must contain "xc", "yc" (nx, ny) cell-centre grids.
    """
    extras = output.extras
    res    = _to_np(output.result)  # (nt+1, nx, ny)
    xc     = _to_np(extras["xc"])
    yc     = _to_np(extras["yc"])
    t_vals = _to_np(extras.get("t", np.arange(res.shape[0]) * extras.get("dt", 1.0)))

    snap = res[frame]
    t_label = f"t={t_vals[frame]:.4g}" if len(t_vals) > abs(frame) else ""
    return plot_scalar(
        xc.ravel(), yc.ravel(), snap.ravel(),
        title=f"{title}  |  {t_label}", cmap=cmap, show=show,
    )


# ---------------------------------------------------------------------------
# Residual dashboard
# ---------------------------------------------------------------------------

def plot_residuals(
    output,  # SolverOutput
    *,
    title: str = "Solver residuals",
    show: bool = False,
) -> Figure:
    """Bar chart of named residual/loss values stored in output.losses."""
    losses = {k: float(_to_np(v)) for k, v in output.losses.items()}
    fig, ax = make_figure(figsize=(max(5, len(losses)), 4))
    bars = ax.bar(list(losses.keys()), list(losses.values()), color="#4c9be8", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log") if any(v > 0 for v in losses.values()) else None
    ax.set_title(title)
    ax.set_ylabel("value")
    for bar, val in zip(bars, losses.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{val:.2e}", ha="center", va="bottom", fontsize=8)
    if show:
        plt.show()
    return fig
