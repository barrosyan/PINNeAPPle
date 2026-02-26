"""Preprocessing pipeline and standard scaler for training batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch


class PreprocessStep(Protocol):
    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "PreprocessStep": ...
    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]: ...


@dataclass
class PreprocessPipeline:
    steps: List[PreprocessStep]

    def fit(self, train_batches: Sequence[Dict[str, Any]]) -> "PreprocessPipeline":
        for s in self.steps:
            s.fit(train_batches)
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = batch
        for s in self.steps:
            out = s.apply(out)
        return out

    def state_dict(self) -> Dict[str, Any]:
        packed = []
        for step in self.steps:
            st = {}
            if hasattr(step, "state_dict"):
                st = step.state_dict()  # type: ignore
            packed.append({"cls": step.__class__.__name__, "state": st})
        return {"steps": packed}

    def load_state_dict(self, state: Dict[str, Any]) -> "PreprocessPipeline":
        packed = (state or {}).get("steps", [])
        for step, item in zip(self.steps, packed):
            if hasattr(step, "load_state_dict"):
                step.load_state_dict(item.get("state", {}))  # type: ignore
        return self


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "eps": float(self.eps)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StandardScaler":
        return StandardScaler(mean=d["mean"], std=d["std"], eps=float(d.get("eps", 1e-8)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    @staticmethod
    def fit(x: torch.Tensor, dim: int | tuple[int, ...] = 0) -> "StandardScaler":
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return StandardScaler(mean=mean, std=std)


@dataclass
class NormalizeStep:
    """
    Fits a StandardScaler on `key` from train batches, then normalizes in apply().
    Works for typical (B,T,D) or (B,D) tensors.
    """
    key: str = "x"
    dim: int | tuple[int, ...] = (0, 1)  # mean/std over batch/time by default
    store_key: str = "normalizer"
    enabled: bool = True

    scaler: Optional[StandardScaler] = None

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "NormalizeStep":
        if not self.enabled:
            return self
        xs = []
        for b in batch_list:
            x = b[self.key]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            xs.append(x)
        x_all = torch.cat(xs, dim=0) if xs[0].ndim >= 2 else torch.stack(xs, dim=0)
        self.scaler = StandardScaler.fit(x_all, dim=self.dim)
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if (not self.enabled) or (self.scaler is None):
            return batch
        out = dict(batch)
        x = out[self.key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        out[self.key] = self.scaler.encode(x)
        out[self.store_key] = self.scaler
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "dim": self.dim,
            "store_key": self.store_key,
            "enabled": bool(self.enabled),
            "scaler": None if self.scaler is None else self.scaler.to_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "NormalizeStep":
        self.key = state.get("key", self.key)
        self.dim = state.get("dim", self.dim)
        self.store_key = state.get("store_key", self.store_key)
        self.enabled = bool(state.get("enabled", self.enabled))
        sc = state.get("scaler", None)
        self.scaler = None if sc is None else StandardScaler.from_dict(sc)
        return self


@dataclass
class SolverFeatureStep:
    solver: Any
    mode: str = "append"
    select_var_dim: Optional[int] = None
    reduce_fft_to: str = "magnitude"
    fit_noop: bool = True

    def fit(self, batch_list):
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch2 = dict(batch)
        x = batch2["x"]

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # --- support x as (T,D) or (B,T,D)
        x_was_2d = (x.ndim == 2)
        if x_was_2d:
            x3 = x.unsqueeze(0)  # (1,T,D)
            squeeze = True
        elif x.ndim == 3:
            x3 = x
            squeeze = False
        else:
            raise ValueError(f"SolverFeatureStep expects x with ndim 2 or 3, got shape {tuple(x.shape)}")

        # --- choose signal: (B,T)
        if self.select_var_dim is not None:
            d = int(self.select_var_dim)
            sig = x3[:, :, d]
        else:
            sig = x3.mean(dim=-1)

        # solver expects (B,T) -> out.result
        out = self.solver(sig)
        feat = out.result
        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat)

        # --- normalize feat to (B,T,F)
        if feat.ndim == 1:
            if feat.shape[0] == x3.shape[1]:
                feat3 = feat[None, :, None]  # (1,T,1)
            else:
                feat3 = feat[:, None, None]  # (B,1,1)
        elif feat.ndim == 2:
            if feat.shape[0] == x3.shape[0] and feat.shape[1] == x3.shape[1]:
                feat3 = feat[:, :, None]     # (B,T,1)
            else:
                feat3 = feat[:, None, :]     # (B,1,F)
        elif feat.ndim == 3:
            feat3 = feat
        else:
            raise ValueError(f"Unsupported solver feature shape: {tuple(feat.shape)}")

        # --- align time length if needed
        if feat3.shape[1] != x3.shape[1]:
            if feat3.shape[1] == 1:
                feat3 = feat3.repeat(1, x3.shape[1], 1)
            else:
                T = min(feat3.shape[1], x3.shape[1])
                feat3 = feat3[:, :T, :]
                x3 = x3[:, :T, :]

        if self.mode == "append":
            x_new = torch.cat([x3, feat3], dim=-1)
        elif self.mode == "replace":
            x_new = feat3
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if squeeze:
            x_new = x_new.squeeze(0)

        batch2["x"] = x_new
        return batch2


@dataclass
class MissingValueStep:
    """Fill missing values (NaNs) in x.

    Supported strategies:
      - 'ffill': forward fill along the time axis
      - 'bfill': backward fill along the time axis
      - 'zero': replace NaNs with 0
    """

    key: str = "x"
    strategy: str = "ffill"
    enabled: bool = True

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "MissingValueStep":
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return batch
        out = dict(batch)
        x = out.get(self.key)
        if x is None:
            return out
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if not torch.isnan(xt).any():
            out[self.key] = xt
            return out

        if self.strategy == "zero":
            xt = torch.nan_to_num(xt, nan=0.0)
        elif self.strategy in ("ffill", "bfill"):
            if xt.ndim == 2:
                xt2 = xt.unsqueeze(0)  # (1,T,D)
                squeeze = True
            elif xt.ndim == 3:
                xt2 = xt  # (B,T,D)
                squeeze = False
            else:
                raise ValueError(f"MissingValueStep expects x with ndim 2 or 3, got {xt.ndim}")

            B, T, D = xt2.shape
            if self.strategy == "ffill":
                for t in range(1, T):
                    mask = torch.isnan(xt2[:, t, :])
                    xt2[:, t, :][mask] = xt2[:, t - 1, :][mask]
            else:
                for t in range(T - 2, -1, -1):
                    mask = torch.isnan(xt2[:, t, :])
                    xt2[:, t, :][mask] = xt2[:, t + 1, :][mask]

            xt = xt2.squeeze(0) if squeeze else xt2
        else:
            raise ValueError(f"Unknown missing strategy: {self.strategy}")

        out[self.key] = xt
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "strategy": self.strategy, "enabled": bool(self.enabled)}

    def load_state_dict(self, state: Dict[str, Any]) -> "MissingValueStep":
        self.key = state.get("key", self.key)
        self.strategy = state.get("strategy", self.strategy)
        self.enabled = bool(state.get("enabled", self.enabled))
        return self


@dataclass
class WinsorizeStep:
    """Quantile clipping (winsorization) fit on the train fold."""

    key: str = "x"
    q_low: float = 0.01
    q_high: float = 0.99
    enabled: bool = True

    low_: Optional[torch.Tensor] = None
    high_: Optional[torch.Tensor] = None

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "WinsorizeStep":
        if not self.enabled:
            return self
        flat: List[torch.Tensor] = []
        for b in batch_list:
            x = b[self.key]
            xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            flat.append(xt.reshape(-1))
        all_x = torch.cat(flat, dim=0)
        self.low_ = torch.quantile(all_x, float(self.q_low))
        self.high_ = torch.quantile(all_x, float(self.q_high))
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if (not self.enabled) or (self.low_ is None) or (self.high_ is None):
            return batch
        out = dict(batch)
        x = out[self.key]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        out[self.key] = torch.clamp(xt, min=self.low_.to(xt.device), max=self.high_.to(xt.device))
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "q_low": float(self.q_low),
            "q_high": float(self.q_high),
            "enabled": bool(self.enabled),
            "low": None if self.low_ is None else self.low_.detach().cpu(),
            "high": None if self.high_ is None else self.high_.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "WinsorizeStep":
        self.key = state.get("key", self.key)
        self.q_low = float(state.get("q_low", self.q_low))
        self.q_high = float(state.get("q_high", self.q_high))
        self.enabled = bool(state.get("enabled", self.enabled))
        self.low_ = state.get("low", None)
        self.high_ = state.get("high", None)
        return self


@dataclass
class RobustScaleStep:
    """Robust scaling: (x - median) / (IQR + eps), fit on the train fold."""

    key: str = "x"
    eps: float = 1e-8
    enabled: bool = True

    median_: Optional[torch.Tensor] = None
    iqr_: Optional[torch.Tensor] = None

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "RobustScaleStep":
        if not self.enabled:
            return self
        flat: List[torch.Tensor] = []
        for b in batch_list:
            x = b[self.key]
            xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            flat.append(xt.reshape(-1))
        all_x = torch.cat(flat, dim=0)
        q25 = torch.quantile(all_x, 0.25)
        q50 = torch.quantile(all_x, 0.50)
        q75 = torch.quantile(all_x, 0.75)
        self.median_ = q50
        self.iqr_ = (q75 - q25).clamp_min(self.eps)
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if (not self.enabled) or (self.median_ is None) or (self.iqr_ is None):
            return batch
        out = dict(batch)
        x = out[self.key]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        out[self.key] = (xt - self.median_.to(xt.device)) / self.iqr_.to(xt.device)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "eps": float(self.eps),
            "enabled": bool(self.enabled),
            "median": None if self.median_ is None else self.median_.detach().cpu(),
            "iqr": None if self.iqr_ is None else self.iqr_.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "RobustScaleStep":
        self.key = state.get("key", self.key)
        self.eps = float(state.get("eps", self.eps))
        self.enabled = bool(state.get("enabled", self.enabled))
        self.median_ = state.get("median", None)
        self.iqr_ = state.get("iqr", None)
        return self


@dataclass
class LogStep:
    """Optional log transform for non-negative signals (fold-safe)."""

    key: str = "x"
    enabled: bool = False
    clamp_min: float = -1e6

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "LogStep":
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return batch
        out = dict(batch)
        x = out[self.key]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        xt = torch.clamp(xt, min=self.clamp_min)
        out[self.key] = torch.log1p(torch.relu(xt))
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "enabled": bool(self.enabled), "clamp_min": float(self.clamp_min)}

    def load_state_dict(self, state: Dict[str, Any]) -> "LogStep":
        self.key = state.get("key", self.key)
        self.enabled = bool(state.get("enabled", self.enabled))
        self.clamp_min = float(state.get("clamp_min", self.clamp_min))
        return self


@dataclass
class DifferencingStep:
    """Differencing along the time axis to improve stationarity.

    Note: this changes sequence length by -order; your dataset/model must be compatible.
    """

    key: str = "x"
    time_dim: int = -2
    order: int = 1
    enabled: bool = False
    anchor_key: str = "diff_anchor"

    def fit(self, batch_list: Sequence[Dict[str, Any]]) -> "DifferencingStep":
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return batch
        out = dict(batch)
        x = out[self.key]
        xt = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if xt.ndim not in (2, 3):
            raise ValueError(f"DifferencingStep expects x with ndim 2 or 3, got {xt.ndim}")

        out[self.anchor_key] = xt.select(dim=self.time_dim, index=0).detach()

        z = xt
        for _ in range(int(self.order)):
            z = torch.diff(z, n=1, dim=self.time_dim)
        out[self.key] = z
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "time_dim": int(self.time_dim),
            "order": int(self.order),
            "enabled": bool(self.enabled),
            "anchor_key": self.anchor_key,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "DifferencingStep":
        self.key = state.get("key", self.key)
        self.time_dim = int(state.get("time_dim", self.time_dim))
        self.order = int(state.get("order", self.order))
        self.enabled = bool(state.get("enabled", self.enabled))
        self.anchor_key = state.get("anchor_key", self.anchor_key)
        return self