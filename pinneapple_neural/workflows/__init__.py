"""End-to-end train/predict workflows over PINNeAPPle's neural
architectures -- reusable, product-facing entry points (as opposed to
``pinneapple_neural.architectures``/``trainer``, which expose the
lower-level building blocks these workflows compose)."""
from __future__ import annotations

from pinneapple_neural.workflows.dense_volume_operator import (
    DenseVolumeOperatorConfig,
    train_dense_volume_operator,
    rollout_dense_volume_operator,
    predict_single_step,
    load_dense_volume_operator,
)

__all__ = [
    "DenseVolumeOperatorConfig",
    "train_dense_volume_operator",
    "rollout_dense_volume_operator",
    "predict_single_step",
    "load_dense_volume_operator",
]
