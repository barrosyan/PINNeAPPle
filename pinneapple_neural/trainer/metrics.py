"""Metrics for evaluation (MSE, MAE, RMSE, R2, MetricBundle)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class Metric:
    name: str
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        raise NotImplementedError


@dataclass
class MSE(Metric):
    name: str = "mse"
    def __call__(self, y_hat, y) -> float:
        return float(torch.mean((y_hat - y) ** 2).item())


@dataclass
class MAE(Metric):
    name: str = "mae"
    def __call__(self, y_hat, y) -> float:
        return float(torch.mean(torch.abs(y_hat - y)).item())


@dataclass
class RMSE(Metric):
    name: str = "rmse"
    def __call__(self, y_hat, y) -> float:
        mse = torch.mean((y_hat - y) ** 2)
        return float(torch.sqrt(mse).item())


@dataclass
class R2(Metric):
    name: str = "r2"
    eps: float = 1e-12
    def __call__(self, y_hat, y) -> float:
        yh = y_hat.reshape(-1, y_hat.shape[-1])
        yt = y.reshape(-1, y.shape[-1])
        y_mean = torch.mean(yt, dim=0, keepdim=True)
        ss_tot = torch.sum((yt - y_mean) ** 2)
        ss_res = torch.sum((yt - yh) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + self.eps))
        return float(r2.mean().item())


@dataclass
class RelL2(Metric):
    name: str = "rel_l2"
    eps: float = 1e-12
    def __call__(self, y_hat, y) -> float:
        num = torch.norm(y_hat - y)
        den = torch.norm(y) + self.eps
        return float((num / den).item())


@dataclass
class MaxError(Metric):
    name: str = "max_error"
    def __call__(self, y_hat, y) -> float:
        return float((y_hat - y).abs().max().item())


def default_metrics() -> List[Metric]:
    return [MSE(), MAE(), RMSE(), R2(), RelL2(), MaxError()]


class Metrics:
    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        raise NotImplementedError


@dataclass
class MetricBundle(Metrics):
    metrics: List[Metric]

    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        return {m.name: float(m(y_hat, y)) for m in self.metrics}

@dataclass
class RegressionMetrics(Metrics):
    eps: float = 1e-12

    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        # flatten all but last dim
        yh = y_hat.reshape(-1, y_hat.shape[-1])
        yt = y.reshape(-1, y.shape[-1])

        mse = torch.mean((yh - yt) ** 2).item()
        mae = torch.mean(torch.abs(yh - yt)).item()
        rmse = float(mse ** 0.5)

        y_mean = torch.mean(yt, dim=0, keepdim=True)
        ss_tot = torch.sum((yt - y_mean) ** 2).item()
        ss_res = torch.sum((yt - yh) ** 2).item()
        r2 = 1.0 - ss_res / max(ss_tot, self.eps)

        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": float(r2)}

def regression_metrics_bundle() -> Metrics:

    return MetricBundle(metrics=[MSE(), MAE(), RMSE(), R2()])
