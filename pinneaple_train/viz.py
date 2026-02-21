"""Visualization utilities for training history plots."""
from __future__ import annotations

from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def plot_history(history: List[Dict[str, float]], keys=("train_total", "val_total")):
    xs = [int(h["epoch"]) for h in history]
    for k in keys:
        ys = [h.get(k) for h in history if k in h]
        if ys:
            plt.plot(xs[:len(ys)], ys, label=k)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("Training history")
    plt.show()
