from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class Split:
    """A single backtesting fold expressed as indices of the windowed dataset."""
    fold: int
    train_idx: List[int]
    val_idx: List[int]
    test_idx: Optional[List[int]] = None


@dataclass
class ExpandingWindowSplitter:
    """Walk-forward validation with an expanding training window.

    Indices refer to items of the *windowed dataset* (not raw timestamps).
    """
    initial_train_size: int
    val_size: int
    step_size: int = 1
    gap: int = 0
    max_folds: Optional[int] = None

    def split(self, n_samples: int) -> Iterator[Split]:
        fold = 0
        train_end = int(self.initial_train_size)

        while True:
            val_start = train_end + int(self.gap)
            val_end = val_start + int(self.val_size)
            if val_end > n_samples:
                return

            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))
            yield Split(fold=fold, train_idx=train_idx, val_idx=val_idx)

            fold += 1
            if self.max_folds is not None and fold >= int(self.max_folds):
                return
            train_end += int(self.step_size)


@dataclass
class RollingWindowSplitter:
    """Walk-forward validation with a fixed-size rolling training window."""
    train_size: int
    val_size: int
    step_size: int = 1
    gap: int = 0
    max_folds: Optional[int] = None

    def split(self, n_samples: int) -> Iterator[Split]:
        fold = 0
        train_start = 0
        train_end = train_start + int(self.train_size)

        while True:
            val_start = train_end + int(self.gap)
            val_end = val_start + int(self.val_size)
            if val_end > n_samples:
                return

            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))
            yield Split(fold=fold, train_idx=train_idx, val_idx=val_idx)

            fold += 1
            if self.max_folds is not None and fold >= int(self.max_folds):
                return
            train_start += int(self.step_size)
            train_end += int(self.step_size)