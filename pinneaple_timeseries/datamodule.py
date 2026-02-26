"""Time series DataModule with deterministic, time-ordered splits and index-based loaders.

This module intentionally avoids random splits (a common source of leakage for forecasting).
It provides helper methods to build DataLoaders from explicit index lists, enabling
walk-forward / rolling / expanding-window validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from .datasets.windowed import WindowedTimeSeriesDataset
from .spec import TimeSeriesSpec


@dataclass
class TSDataModule:
    series: torch.Tensor
    spec: TimeSeriesSpec
    batch_size: int = 64
    val_ratio: float = 0.2
    num_workers: int = 0
    pin_memory: bool = True

    def dataset(self) -> WindowedTimeSeriesDataset:
        return WindowedTimeSeriesDataset(self.series, self.spec)

    def make_loaders_by_indices(
        self,
        train_idx: Sequence[int],
        val_idx: Sequence[int],
        *,
        shuffle_train: bool = True,
        val_batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        ds = self.dataset()
        train_ds = Subset(ds, list(train_idx))
        val_ds = Subset(ds, list(val_idx))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=bool(shuffle_train),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(val_batch_size or max(128, self.batch_size)),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader

    def make_sequential_holdout_loaders(
        self,
        *,
        val_ratio: Optional[float] = None,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Deterministic holdout split: first train windows, last validation windows."""
        ds = self.dataset()
        n = len(ds)
        vr = float(self.val_ratio if val_ratio is None else val_ratio)
        n_val = int(n * vr)
        n_train = n - n_val
        train_idx = list(range(0, n_train))
        val_idx = list(range(n_train, n))
        return self.make_loaders_by_indices(train_idx, val_idx, shuffle_train=shuffle_train)

    # Backward-compatible alias
    def make_loaders(self):
        """Backward-compatible API: returns deterministic holdout loaders."""
        return self.make_sequential_holdout_loaders(val_ratio=self.val_ratio, shuffle_train=True)