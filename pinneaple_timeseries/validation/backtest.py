from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch.nn as nn

from pinneaple_train.trainer import TrainConfig, Trainer
from pinneaple_train.preprocess import PreprocessPipeline

from ..datamodule import TSDataModule
from .splitters import Split


@dataclass
class BacktestConfig:
    """Backtesting settings."""
    reset_model_each_fold: bool = True
    shuffle_train: bool = True
    val_batch_size: Optional[int] = None


@dataclass
class BacktestResult:
    folds: List[Dict[str, Any]]
    agg: Dict[str, float]


def _mean(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


class BacktestRunner:
    """Train/evaluate a model across a sequence of time-series splits."""

    def __init__(
        self,
        *,
        trainer_factory,
        model_factory,
        datamodule: TSDataModule,
        splits: Sequence[Split],
        train_cfg: TrainConfig,
        preprocess_factory=None,
        backtest_cfg: Optional[BacktestConfig] = None,
    ):
        """Create a backtest runner.

        trainer_factory: (model, preprocess) -> Trainer
        model_factory: () -> nn.Module
        preprocess_factory: () -> PreprocessPipeline | None
        """
        self.trainer_factory = trainer_factory
        self.model_factory = model_factory
        self.preprocess_factory = preprocess_factory
        self.datamodule = datamodule
        self.splits = list(splits)
        self.train_cfg = train_cfg
        self.bt_cfg = backtest_cfg or BacktestConfig()

    def run(self) -> BacktestResult:
        ds = self.datamodule.dataset()
        n = len(ds)

        fold_records: List[Dict[str, Any]] = []
        last_model_state = None

        for split in self.splits:
            if max(split.train_idx + split.val_idx) >= n:
                raise ValueError("Split indices exceed dataset length.")

            model: nn.Module = self.model_factory()
            if (not self.bt_cfg.reset_model_each_fold) and (last_model_state is not None):
                model.load_state_dict(last_model_state)

            preprocess: Optional[PreprocessPipeline] = self.preprocess_factory() if self.preprocess_factory else None
            trainer: Trainer = self.trainer_factory(model=model, preprocess=preprocess)

            train_loader, val_loader = self.datamodule.make_loaders_by_indices(
                split.train_idx,
                split.val_idx,
                shuffle_train=self.bt_cfg.shuffle_train,
                val_batch_size=self.bt_cfg.val_batch_size,
            )

            # Preprocessing is fit inside Trainer.fit() on the train fold only.
            out = trainer.fit(train_loader, val_loader, cfg=self.train_cfg)

            record: Dict[str, Any] = {
                "fold": split.fold,
                "best_val": out.get("best_val"),
                "best_path": out.get("best_path"),
            }

            history = out.get("history") or []
            if history:
                record.update({k: v for k, v in history[-1].items() if isinstance(v, (int, float))})

            fold_records.append(record)
            last_model_state = model.state_dict()

        agg: Dict[str, float] = {}
        for k in ["best_val", "val_total", "val_mae", "val_rmse", "val_smape", "val_mase"]:
            m = _mean(fold_records, k)
            if m is not None:
                agg[f"mean_{k}"] = m

        return BacktestResult(folds=fold_records, agg=agg)