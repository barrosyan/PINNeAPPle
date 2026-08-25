"""
Generic model-evaluation plots: error/value histograms, real-vs-predicted
scatter, calibration curve. Pure matplotlib, no framework/storage coupling —
callers pass arrays and an output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_value_histograms(
    fields: Dict[str, np.ndarray],
    out_dir: Union[str, Path],
    bins: int = 40,
) -> List[str]:
    """One histogram per field of its raw values (use when there's no
    ground-truth comparison, e.g. profiling a dataset rather than scoring a
    model)."""
    out_dir = Path(out_dir)
    plots = []
    for name, arr in fields.items():
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(np.asarray(arr, dtype=float).reshape(-1), bins=bins, color="#4C72B0")
        ax.set_title(f"{name} — value distribution")
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        plots.append(_fig_save(fig, out_dir / f"hist_{name}.png"))
    return plots


def plot_error_histograms(
    abs_residuals: np.ndarray,
    field_names: Sequence[str],
    out_dir: Union[str, Path],
    bins: int = 40,
) -> List[str]:
    """One histogram per output field of |prediction - ground truth|."""
    out_dir = Path(out_dir)
    abs_residuals = np.asarray(abs_residuals)
    plots = []
    for i, name in enumerate(field_names):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(abs_residuals[:, i], bins=bins, color="#DD8452")
        ax.set_title(f"{name} — |error| distribution")
        ax.set_xlabel("|prediction - ground truth|")
        ax.set_ylabel("count")
        plots.append(_fig_save(fig, out_dir / f"hist_error_{name}.png"))
    return plots


def plot_real_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Union[str, Path],
) -> str:
    """Scatter of predicted vs. actual values with a y=x reference line."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#4C72B0")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Real vs. Predicted")
    return _fig_save(fig, Path(out_path))


def plot_calibration_curve(
    levels: Sequence[float],
    empirical_coverage: Sequence[float],
    out_path: Union[str, Path],
) -> str:
    """Empirical vs. nominal confidence coverage, against the ideal diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(levels, levels, "k--", label="ideal")
    ax.plot(levels, empirical_coverage, "o-", color="#55A868", label="empirical")
    ax.set_xlabel("nominal confidence")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration Curve")
    ax.legend()
    return _fig_save(fig, Path(out_path))
