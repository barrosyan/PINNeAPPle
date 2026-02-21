"""Normalizers: StandardScaler and MinMaxScaler for tensor preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch


class Normalizer:
    def fit(self, x: torch.Tensor) -> "Normalizer":
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, sd: Dict[str, Any]) -> "Normalizer":
        return self


@dataclass
class StandardScaler(Normalizer):
    eps: float = 1e-8
    mean_: Optional[torch.Tensor] = None
    std_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "StandardScaler":
        # x: (..., D) normalize last dim
        self.mean_ = x.mean(dim=0, keepdim=True)
        self.std_ = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            return x
        return (x - self.mean_.to(x.device, x.dtype)) / self.std_.to(x.device, x.dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None or self.std_ is None:
            return x
        return x * self.std_.to(x.device, x.dtype) + self.mean_.to(x.device, x.dtype)

    def state_dict(self):
        return {"mean_": self.mean_, "std_": self.std_, "eps": self.eps}

    def load_state_dict(self, sd):
        self.mean_ = sd.get("mean_")
        self.std_ = sd.get("std_")
        self.eps = float(sd.get("eps", self.eps))
        return self


@dataclass
class MinMaxScaler(Normalizer):
    eps: float = 1e-8
    min_: Optional[torch.Tensor] = None
    max_: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "MinMaxScaler":
        self.min_ = x.amin(dim=0, keepdim=True)
        self.max_ = x.amax(dim=0, keepdim=True)
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

    def state_dict(self):
        return {"min_": self.min_, "max_": self.max_, "eps": self.eps}

    def load_state_dict(self, sd):
        self.min_ = sd.get("min_")
        self.max_ = sd.get("max_")
        self.eps = float(sd.get("eps", self.eps))
        return self
