"""Visualization utilities for training history.

Delegates to ``pinneaple_viz.pinn`` for rich, log-scaled figures.
A thin ``plot_history`` shim is kept here for backwards compatibility.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

try:
    from pinneaple_tools.visualization.pinn import plot_loss_history as _plot_loss_history, plot_multi_loss
    _VIZ_OK = True
except ImportError:  # pragma: no cover
    _VIZ_OK = False


def plot_history(
    history: List[Dict[str, float]],
    keys: Sequence[str] = ("train_total", "val_total"),
    *,
    log_scale: bool = True,
    show: bool = True,
):
    """Plot training history.

    Delegates to :func:`pinneaple_viz.pinn.plot_loss_history` when available
    (returns a ``Figure``). Falls back to a minimal matplotlib plot otherwise.

    Parameters
    ----------
    history : list of dicts with at least ``"epoch"`` and loss keys.
    keys : which keys to plot (default: train_total and val_total).
    log_scale : use log y-axis (default True).
    show : call ``plt.show()`` at the end (default True).
    """
    if _VIZ_OK:
        return _plot_loss_history(history, keys=list(keys), log_scale=log_scale, show=show)

    # Minimal fallback (no pinneaple_viz installed)
    import matplotlib.pyplot as plt
    xs = [int(h.get("epoch", i)) for i, h in enumerate(history)]
    for k in keys:
        ys = [h[k] for h in history if k in h]
        if ys:
            plt.plot(xs[: len(ys)], ys, label=k)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.title("Training history")
    if show:
        plt.show()


__all__ = ["plot_history", "plot_multi_loss"]
