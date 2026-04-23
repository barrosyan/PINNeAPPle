"""PINN-specific visualisations: training curves, residual fields, collocation points."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .style import get_cmap, make_figure, DEFAULT_CMAP
from .fields import plot_scalar, plot_error


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_loss_history(
    history: List[Dict[str, float]],
    *,
    keys: Optional[Sequence[str]] = None,
    log_scale: bool = True,
    title: str = "Training loss history",
    show: bool = False,
) -> Figure:
    """
    Plot loss curves from a training history list.

    history : list of dicts with at least "epoch" and loss keys
    keys    : which loss keys to plot (None = all numeric keys except "epoch")
    """
    if not history:
        raise ValueError("history is empty")

    fig, ax = make_figure(figsize=(9, 4))
    epochs = [h.get("epoch", i) for i, h in enumerate(history)]

    if keys is None:
        keys = [k for k in history[0] if k != "epoch" and isinstance(history[0][k], (int, float))]

    palette = plt.cm.tab10(np.linspace(0, 1, max(len(keys), 1)))
    for i, k in enumerate(keys):
        ys = [h[k] for h in history if k in h and not np.isnan(h[k])]
        xs = epochs[: len(ys)]
        ax.plot(xs, ys, label=k, lw=1.5, color=palette[i])

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_multi_loss(
    history: List[Dict[str, float]],
    *,
    groups: Optional[Dict[str, Sequence[str]]] = None,
    log_scale: bool = True,
    title: str = "Loss components",
    show: bool = False,
) -> Figure:
    """
    Multi-panel loss breakdown.

    groups : {"Physics": ["pde_loss", ...], "Data": ["data_loss"]} — groups into sub-axes.
    """
    if groups is None:
        groups = {"All": [k for k in history[0] if k != "epoch"]}

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    epochs = [h.get("epoch", i) for i, h in enumerate(history)]
    palette = plt.cm.Set2(np.linspace(0, 1, 8))

    for ax, (gname, gkeys) in zip(axes, groups.items()):
        for i, k in enumerate(gkeys):
            ys = [h.get(k, np.nan) for h in history]
            ax.plot(epochs, ys, label=k, lw=1.4, color=palette[i % 8])
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(gname)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Collocation point visualisation
# ---------------------------------------------------------------------------

def plot_collocation(
    interior_pts: "ArrayLike",
    boundary_pts: Optional["ArrayLike"] = None,
    initial_pts: Optional["ArrayLike"] = None,
    data_pts: Optional["ArrayLike"] = None,
    *,
    title: str = "Collocation points",
    show: bool = False,
) -> Figure:
    """
    Scatter plot of collocation point sets in 2-D.

    Colour coding:
      blue   = interior
      red    = boundary
      green  = initial condition
      orange = observed data
    """
    fig, ax = make_figure(figsize=(7, 6))

    def _scatter(pts, color, label, marker="o", s=6):
        if pts is None:
            return
        p = _to_np(pts)
        ax.scatter(p[:, 0], p[:, 1], c=color, s=s, marker=marker,
                   label=label, alpha=0.7, linewidths=0)

    _scatter(interior_pts,  "#4c9be8", "Interior",        s=4)
    _scatter(boundary_pts,  "#e84c4c", "Boundary",        marker="^", s=20)
    _scatter(initial_pts,   "#4ce87a", "Initial cond.",   marker="s", s=12)
    _scatter(data_pts,      "#f0a500", "Observed data",   marker="D", s=15)

    ax.legend(fontsize=9, markerscale=2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# PINN prediction panel
# ---------------------------------------------------------------------------

def plot_pinn_prediction(
    x: "ArrayLike",
    y: "ArrayLike",
    pred: "ArrayLike",
    ref: Optional["ArrayLike"] = None,
    *,
    field_name: str = "u",
    cmap: str = DEFAULT_CMAP,
    show: bool = False,
) -> Figure:
    """
    Two- or three-panel comparison: [Reference] [Prediction] [Error].
    If ref is None, shows only the prediction.
    """
    panels = []
    if ref is not None:
        panels.append((f"Reference ({field_name})", ref))
    panels.append((f"PINN prediction ({field_name})", pred))
    if ref is not None:
        err = np.abs(_to_np(pred).ravel() - _to_np(ref).ravel())
        ref_scale = np.abs(_to_np(ref).ravel()).max() + 1e-12
        panels.append((f"Relative error ({field_name})", err / ref_scale))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    cmaps = [cmap] * (n - 1) + (["hot"] if ref is not None else [cmap])
    for ax, (label, field), cm in zip(axes, panels, cmaps):
        plot_scalar(_to_np(x), _to_np(y), field, title=label, cmap=cm, axes=ax, label=label)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# PDE residual field (model-independent)
# ---------------------------------------------------------------------------

def plot_pde_residual(
    x: "ArrayLike",
    y: "ArrayLike",
    residual: "ArrayLike",
    *,
    title: str = "PDE residual",
    cmap: str = "hot",
    log_scale: bool = True,
    show: bool = False,
) -> Figure:
    """
    Plot |PDE residual| at collocation points.
    Optionally log-scaled for better dynamic range.
    """
    r = np.abs(_to_np(residual).ravel())
    if log_scale:
        r = np.log10(r + 1e-12)
        label = "log₁₀|residual|"
    else:
        label = "|residual|"

    return plot_scalar(_to_np(x), _to_np(y), r, title=title, label=label, cmap=cmap, show=show)


# ---------------------------------------------------------------------------
# Sensitivity / gradient magnitude
# ---------------------------------------------------------------------------

def plot_gradient_magnitude(
    x: "ArrayLike",
    y: "ArrayLike",
    grad_x: "ArrayLike",
    grad_y: "ArrayLike",
    *,
    title: str = "Gradient magnitude |∇u|",
    cmap: str = "plasma",
    show: bool = False,
) -> Figure:
    """Plot |∇u| from separate partial derivative arrays."""
    gx = _to_np(grad_x).ravel()
    gy = _to_np(grad_y).ravel()
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return plot_scalar(_to_np(x), _to_np(y), mag, title=title, label="|∇u|", cmap=cmap, show=show)
