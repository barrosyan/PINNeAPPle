"""Build Metrics objects from YAML-driven configuration lists.

Usage
-----
from pinneaple_train.metrics_cfg import build_metrics_from_cfg

metrics = build_metrics_from_cfg(["mse", "rmse", "rel_l2", "r2"])
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from .metrics import (
    Metric,
    Metrics,
    MetricBundle,
    MSE,
    MAE,
    RMSE,
    R2,
    RelL2,
    MaxError,
    default_metrics,
)

_METRIC_CLASSES: Dict[str, type] = {
    "mse": MSE,
    "mae": MAE,
    "rmse": RMSE,
    "r2": R2,
    "rel_l2": RelL2,
    "max_error": MaxError,
}


def build_metrics_from_cfg(
    cfg: List[Union[str, Dict[str, Any]]],
) -> Metrics:
    """Build a MetricBundle from a list of metric names or config dicts.

    Each entry can be:
    - A string: metric name (e.g. "mse", "rmse", "rel_l2", "r2", "mae", "max_error")
    - A dict with at least a "name" key and optional kwargs

    Parameters
    ----------
    cfg : list of metric specs

    Returns
    -------
    MetricBundle wrapping the requested Metric instances

    Raises
    ------
    KeyError if an unknown metric name is encountered
    """
    if not cfg:
        return MetricBundle(default_metrics())

    metrics: List[Metric] = []
    for entry in cfg:
        if isinstance(entry, str):
            name = entry.strip().lower()
            if name not in _METRIC_CLASSES:
                raise KeyError(
                    f"Unknown metric '{name}'. Available: {sorted(_METRIC_CLASSES.keys())}"
                )
            metrics.append(_METRIC_CLASSES[name]())
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).strip().lower()
            kwargs = {k: v for k, v in entry.items() if k != "name"}
            if name not in _METRIC_CLASSES:
                raise KeyError(
                    f"Unknown metric '{name}'. Available: {sorted(_METRIC_CLASSES.keys())}"
                )
            metrics.append(_METRIC_CLASSES[name](**kwargs))
        else:
            raise TypeError(f"Metric config entry must be str or dict, got {type(entry)}")

    return MetricBundle(metrics=metrics)
