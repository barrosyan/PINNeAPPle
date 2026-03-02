from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    max_points: int = 1024,
) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size > max_points:
        idx = np.linspace(0, y_true.size - 1, max_points).astype(int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig = plt.figure()
    plt.title(title)
    plt.plot(y_true, label="reference")
    plt.plot(y_pred, label="model")
    plt.legend()
    plt.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def plot_heatmaps(
    ref: np.ndarray,
    pred: np.ndarray,
    title: str,
) -> str:
    ref = np.asarray(ref)
    pred = np.asarray(pred)
    err = np.abs(pred - ref)

    fig = plt.figure(figsize=(10, 3.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.set_title("Reference")
    ax2.set_title("Model")
    ax3.set_title("Abs Error")

    im1 = ax1.imshow(ref, origin="lower", aspect="auto")
    im2 = ax2.imshow(pred, origin="lower", aspect="auto")
    im3 = ax3.imshow(err, origin="lower", aspect="auto")

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    return fig_to_base64(fig)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean((a - b) ** 2))


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + eps
    return float(num / den)
