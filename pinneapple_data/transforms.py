"""Data transforms and normalizers for preprocessing pipelines.

Single source of truth for all scaling/normalization classes used across
pinneapple_train and pinneapple_data.  pinneapple_train.normalizers and
pinneapple_train.preprocess both re-export from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


class Normalizer:
    """Abstract base for all normalizers (fit / transform / inverse)."""

    def fit(self, x: torch.Tensor) -> "Normalizer":
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ── aliases used by pinneapple_train.preprocess (encode/decode API) ──────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse(x)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, sd: Dict[str, Any]) -> "Normalizer":
        return self


@dataclass
class StandardScaler(Normalizer):
    """Zero-mean unit-variance scaler.

    Supports two construction styles:
      - ``StandardScaler.fit(x)``          ← class method, stateless style
      - ``StandardScaler().fit(x)``        ← instance style (pinneapple_train)

    Both ``transform``/``inverse`` and ``encode``/``decode`` work identically.
    ``to_dict`` / ``from_dict`` preserve state for serialization.
    """

    eps: float = 1e-8
    mean_: Optional[torch.Tensor] = None
    std_: Optional[torch.Tensor] = None

    # ── instance fit (pinneapple_train.normalizers style) ────────────────────
    def fit(self, x: torch.Tensor, dim: int | tuple = 0) -> "StandardScaler":  # type: ignore[override]
        self.mean_ = x.mean(dim=dim, keepdim=True)
        self.std_ = x.std(dim=dim, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            return x
        return (x - self.mean_.to(x.device, x.dtype)) / self.std_.to(x.device, x.dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            return x
        return x * self.std_.to(x.device, x.dtype) + self.mean_.to(x.device, x.dtype)

    # ── serialization (pinneapple_train.preprocess style) ────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.detach().cpu() if self.mean_ is not None else None,
            "std": self.std_.detach().cpu() if self.std_ is not None else None,
            "eps": float(self.eps),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StandardScaler":
        s = StandardScaler(eps=float(d.get("eps", 1e-8)))
        s.mean_ = d.get("mean")
        s.std_ = d.get("std")
        return s

    def state_dict(self) -> Dict[str, Any]:
        return self.to_dict()

    def load_state_dict(self, sd: Dict[str, Any]) -> "StandardScaler":
        self.mean_ = sd.get("mean_") or sd.get("mean")
        self.std_ = sd.get("std_") or sd.get("std")
        self.eps = float(sd.get("eps", self.eps))
        return self


@dataclass
class MinMaxScaler(Normalizer):
    """Scales each feature to [0, 1]."""

    eps: float = 1e-8
    min_: Optional[torch.Tensor] = None
    max_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor, dim: int | tuple = 0) -> "MinMaxScaler":  # type: ignore[override]
        self.min_ = x.amin(dim=dim, keepdim=True)
        self.max_ = x.amax(dim=dim, keepdim=True)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_ is None or self.max_ is None:
            return x
        mn = self.min_.to(x.device, x.dtype)
        mx = self.max_.to(x.device, x.dtype)
        return (x - mn) / (mx - mn).clamp_min(self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_ is None or self.max_ is None:
            return x
        mn = self.min_.to(x.device, x.dtype)
        mx = self.max_.to(x.device, x.dtype)
        return x * (mx - mn) + mn

    def state_dict(self) -> Dict[str, Any]:
        return {
            "min_": self.min_.detach().cpu() if self.min_ is not None else None,
            "max_": self.max_.detach().cpu() if self.max_ is not None else None,
            "eps": float(self.eps),
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> "MinMaxScaler":
        self.min_ = sd.get("min_")
        self.max_ = sd.get("max_")
        self.eps = float(sd.get("eps", self.eps))
        return self


__all__ = ["Normalizer", "StandardScaler", "MinMaxScaler"]
