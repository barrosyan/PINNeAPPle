"""Compare multiple trained model runs.

This is an Arena-level utility that:
  - loads saved predictions (or checkpoints)
  - runs inference on a shared test set
  - computes metrics
  - writes .json/.csv tables
  - generates basic plots

MVP: works with pointwise coordinate datasets (x->y).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y) ** 2))


def _l2(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.sum((y_hat - y) ** 2)))


def _rel_l2(y_hat: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sqrt(np.sum((y_hat - y) ** 2))
    den = np.sqrt(np.sum(y**2)) + eps
    return float(num / den)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _try_load_prediction(run_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """If run saved preds/gt, load them."""
    pred_path = run_dir / "predictions.npz"
    if not pred_path.exists():
        return None
    d = np.load(pred_path)
    if "y_hat" in d and "y" in d:
        return d["y_hat"], d["y"]
    return None


def compare_runs(
    run_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path = "runs/arena_compare",
    title: str = "arena_compare",
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Compare completed run directories.

    Each run_dir must contain either:
      - predictions.npz with arrays y_hat and y
      - or (future) a checkpoint + config (not implemented in this MVP)
    """

    out_dir = _ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        run_dir = Path(rd)
        name = run_dir.name
        pair = _try_load_prediction(run_dir)
        if pair is None:
            raise RuntimeError(
                f"Run '{name}' is missing predictions.npz. "
                "MVP compare expects run_benchmark to export it."
            )
        y_hat, y = pair
        rows.append(
            {
                "model": name,
                "mse": _mse(y_hat, y),
                "l2": _l2(y_hat, y),
                "rel_l2": _rel_l2(y_hat, y),
                "n": int(y.shape[0]),
                "d": int(y.shape[1]) if y.ndim == 2 else 1,
            }
        )

    # ranking
    rows_sorted = sorted(rows, key=lambda r: (r["rel_l2"], r["mse"]))
    for i, r in enumerate(rows_sorted, start=1):
        r["rank"] = i

    _write_json(out_dir / f"{title}.json", {"rows": rows_sorted})
    _write_csv(out_dir / f"{title}.csv", rows_sorted)

    if make_plots:
        _make_basic_plots(run_dirs, out_dir / f"{title}_plots")

    return {"rows": rows_sorted, "out_dir": str(out_dir)}


def _make_basic_plots(run_dirs: Sequence[str | Path], out_dir: Path) -> None:
    """Generate MVP plots.

    If output is 1D or 2D, we create basic plots without assuming a mesh.
    """
    out_dir = _ensure_dir(out_dir)

    import matplotlib.pyplot as plt

    # plot error distribution per model
    for rd in run_dirs:
        run_dir = Path(rd)
        name = run_dir.name
        pair = _try_load_prediction(run_dir)
        if pair is None:
            continue
        y_hat, y = pair
        err = np.linalg.norm((y_hat - y).reshape(y.shape[0], -1), axis=1)

        plt.figure()
        plt.hist(err, bins=50)
        plt.title(f"Error histogram: {name}")
        plt.xlabel("L2 error per sample")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_err_hist.png", dpi=150)
        plt.close()
