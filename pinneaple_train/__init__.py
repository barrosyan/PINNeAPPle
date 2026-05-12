"""pinneaple_train — compatibility shim.

Re-exports from pinneaple_neural.trainer so that legacy code importing
``from pinneaple_train.*`` continues to work.

All new code should import directly from ``pinneaple_neural.trainer``.
"""
from pinneaple_neural.trainer.trainer import (
    Trainer,
    TrainConfig,
)
from pinneaple_neural.trainer.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pinneaple_neural.trainer.metrics import (
    Metrics,
    Metric,
    MetricBundle,
    default_metrics,
)

try:
    from pinneaple_neural.trainer.time_marching import TimeMarchingTrainer
except Exception:
    TimeMarchingTrainer = None  # type: ignore

try:
    from pinneaple_neural.trainer.audit import RunLogger
except Exception:
    RunLogger = None  # type: ignore

try:
    from pinneaple_neural.trainer.checkpoint import Checkpoint, save_checkpoint
except Exception:
    Checkpoint = None           # type: ignore
    save_checkpoint = None      # type: ignore

__all__ = [
    "Trainer", "TrainConfig",
    "EarlyStopping", "ModelCheckpoint",
    "Metrics", "Metric", "MetricBundle", "default_metrics",
    "TimeMarchingTrainer",
    "RunLogger",
    "Checkpoint", "save_checkpoint",
]
