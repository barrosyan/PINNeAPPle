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


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-8

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
        elif x.ndim == 3:
            x3 = x
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
            # (T,) or (B,) is ambiguous. If matches T, treat as (T,)
            if feat.shape[0] == x3.shape[1]:
                feat3 = feat[None, :, None]  # (1,T,1)
            else:
                # treat as (B,) -> (B,1,1) then expand over T
                feat3 = feat[:, None, None].expand(-1, x3.shape[1], -1)
        elif feat.ndim == 2:
            # could be (B,T) or (T,F) or (B,F)
            if feat.shape[0] == x3.shape[0] and feat.shape[1] == x3.shape[1]:
                feat3 = feat[:, :, None]  # (B,T,1)
            elif feat.shape[0] == x3.shape[1]:
                # (T,F) -> (B,T,F)
                feat3 = feat[None, :, :].expand(x3.shape[0], -1, -1)
            elif feat.shape[0] == x3.shape[0]:
                # (B,F) -> (B,T,F)
                feat3 = feat[:, None, :].expand(-1, x3.shape[1], -1)
            else:
                raise ValueError(
                    f"Cannot align feat shape {tuple(feat.shape)} with x shape {tuple(x3.shape)}"
                )
        elif feat.ndim == 3:
            feat3 = feat
        else:
            raise ValueError(f"Unsupported feat ndim={feat.ndim} shape={tuple(feat.shape)}")

        # reduce FFT complex-like last dim=2 => magnitude (keep 1 channel)
        if feat3.ndim == 3 and feat3.shape[-1] == 2 and self.reduce_fft_to == "magnitude":
            feat3 = feat3[..., 0:1]

        # match device/dtype
        feat3 = feat3.to(device=x3.device, dtype=x3.dtype)

        batch2["x_features"] = feat3.squeeze(0) if x_was_2d else feat3

        if self.mode == "append":
            # time-align
            T = x3.shape[1]
            if feat3.shape[1] != T:
                if feat3.shape[1] < T:
                    pad = torch.zeros(
                        (feat3.shape[0], T - feat3.shape[1], feat3.shape[2]),
                        device=feat3.device,
                        dtype=feat3.dtype,
                    )
                    feat3 = torch.cat([feat3, pad], dim=1)
                else:
                    feat3 = feat3[:, :T, :]

            x_out = torch.cat([x3, feat3], dim=-1)
            batch2["x"] = x_out.squeeze(0) if x_was_2d else x_out

        elif self.mode == "replace":
            batch2["x"] = feat3.squeeze(0) if x_was_2d else feat3
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use 'append' or 'replace'.")

        return batch2
