"""Training callbacks: EarlyStopping and ModelCheckpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch


@dataclass
class EarlyStopping:
    monitor: str = "val_total"
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"  # "min" or "max"
    best: Optional[float] = None
    bad_epochs: int = 0
    stop: bool = False

    def update(self, logs: Dict[str, float]) -> None:
        cur = logs.get(self.monitor)
        if cur is None:
            return

        improved = False
        if self.best is None:
            improved = True
        else:
            if self.mode == "min":
                improved = (cur < self.best - self.min_delta)
            else:
                improved = (cur > self.best + self.min_delta)

        if improved:
            self.best = float(cur)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stop = True


@dataclass
class ModelCheckpoint:
    path: str
    monitor: str = "val_total"
    mode: str = "min"
    best: Optional[float] = None

    def maybe_save(self, model: torch.nn.Module, extra: Dict[str, Any], logs: Dict[str, float]) -> bool:
        cur = logs.get(self.monitor)
        if cur is None:
            return False

        improved = False
        if self.best is None:
            improved = True
        else:
            improved = (cur < self.best) if self.mode == "min" else (cur > self.best)

        if improved:
            self.best = float(cur)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save({"model": model.state_dict(), "extra": extra, "best": self.best}, self.path)
            return True
        return False
