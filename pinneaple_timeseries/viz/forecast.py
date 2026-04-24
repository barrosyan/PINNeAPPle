"""Forecast visualization: rolling predictions, confidence bands, parity plots."""
from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("pip install matplotlib")


# ---------------------------------------------------------------------------
# Rolling 1-step-ahead forecast with uncertainty bands
# ---------------------------------------------------------------------------

def plot_rolling_forecast(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_std:   Optional[np.ndarray] = None,
    sigmas:  Sequence[float] = (1.0, 2.0, 3.0),
    title:   str = "Rolling Forecast",
    xlabel:  str = "Time step",
    ylabel:  str = "Value",
    figsize: tuple = (14, 5),
    ax=None,
):
    """
    Plot actual vs. predicted with ±1σ/2σ/3σ confidence bands.

    Parameters
    ----------
    y_true  : 1-D array of ground truth values
    y_pred  : 1-D array of point forecasts (same length as y_true)
    y_std   : 1-D array of predicted standard deviations (optional)
    sigmas  : sigma multiples for confidence bands
    """
    plt = _ensure_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    t = np.arange(len(y_true))
    ax.plot(t, y_true, color="#2196F3", lw=1.5, label="Actual")
    ax.plot(t, y_pred, color="#FF5722", lw=1.5, ls="--", label="Forecast")

    if y_std is not None:
        alphas = [0.25, 0.15, 0.07]
        colors = ["#FF5722"] * 3
        for sigma, alpha in zip(sorted(sigmas, reverse=True), alphas):
            ax.fill_between(
                t,
                y_pred - sigma * y_std,
                y_pred + sigma * y_std,
                alpha=alpha, color=colors[0],
                label=f"±{sigma:.0f}σ",
            )

    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-horizon forecast plot
# ---------------------------------------------------------------------------

def plot_forecast_horizon(
    y_context: np.ndarray,
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_std:     Optional[np.ndarray] = None,
    sigmas:    Sequence[float] = (1.0, 2.0),
    title:     str = "Forecast",
    figsize:   tuple = (12, 4),
):
    """Plot context + future ground truth + forecast with confidence bands."""
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    t_ctx  = np.arange(len(y_context))
    t_fwd  = np.arange(len(y_context), len(y_context) + len(y_pred))

    ax.plot(t_ctx, y_context, color="#78909C", lw=1.2, label="Context")
    ax.plot(t_fwd, y_true,   color="#2196F3", lw=1.5, label="True future")
    ax.plot(t_fwd, y_pred,   color="#FF5722", lw=1.5, ls="--", label="Forecast")
    ax.axvline(x=len(y_context) - 0.5, color="gray", ls=":", lw=1)

    if y_std is not None:
        alphas = [0.25, 0.12]
        for sigma, alpha in zip(sorted(sigmas, reverse=True), alphas):
            ax.fill_between(t_fwd, y_pred - sigma * y_std, y_pred + sigma * y_std,
                            alpha=alpha, color="#FF5722", label=f"±{sigma:.0f}σ")

    ax.set_title(title); ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title:  str = "Parity Plot",
    figsize: tuple = (5, 5),
    ax=None,
):
    """Scatter plot of actual vs. predicted with diagonal reference line."""
    plt = _ensure_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    ax.scatter(y_true, y_pred, s=8, alpha=0.4, color="#1976D2")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    title:   str = "Residual Analysis",
    figsize: tuple = (14, 4),
):
    """Three-panel: residuals vs time, histogram, Q-Q plot."""
    plt    = _ensure_mpl()
    from scipy import stats as sp_stats
    res    = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    t      = np.arange(len(res))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Time series of residuals
    axes[0].plot(t, res, lw=0.8, color="#37474F")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_title("Residuals"); axes[0].grid(True, alpha=0.3)

    # Histogram
    axes[1].hist(res, bins=30, color="#1565C0", alpha=0.7, edgecolor="white")
    axes[1].set_title("Residual Distribution"); axes[1].grid(True, alpha=0.3)

    # Q-Q
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(res, dist="norm")
    axes[2].scatter(osm, osr, s=6, alpha=0.5, color="#1976D2")
    x_line = np.array([min(osm), max(osm)])
    axes[2].plot(x_line, slope * x_line + intercept, "r--", lw=1.5)
    axes[2].set_title(f"Q-Q Plot (r={r:.3f})"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Backtesting results
# ---------------------------------------------------------------------------

def plot_backtest(
    y_true:      np.ndarray,
    predictions: dict,
    title:       str = "Backtest Comparison",
    figsize:     tuple = (14, 5),
):
    """
    Overlay multiple model predictions on ground truth.

    Parameters
    ----------
    predictions : dict[str → np.ndarray]  model_name → 1-D predictions
    """
    plt = _ensure_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    t = np.arange(len(y_true))
    ax.plot(t, y_true, color="#212121", lw=2, label="Actual", zorder=10)

    colors = plt.cm.tab10.colors
    for i, (name, yp) in enumerate(predictions.items()):
        ax.plot(t[:len(yp)], yp, color=colors[i % 10], lw=1.2, ls="--", label=name)

    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
