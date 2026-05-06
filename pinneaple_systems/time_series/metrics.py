"""Time series metrics for forecasting evaluation.

All functions accept (y_hat, batch) where batch[1] is the target tensor,
matching the pinneaple_timeseries training loop convention.
Standalone variants (prefix _fn) accept (y_hat, y) tensors directly.
"""
import torch


# ── Batch-API metrics ─────────────────────────────────────────────────────────

def mae(y_hat, batch):
    """Mean Absolute Error."""
    y = batch[1]
    return torch.mean(torch.abs(y_hat - y))


def mse(y_hat, batch):
    """Mean Squared Error."""
    y = batch[1]
    return torch.mean((y_hat - y) ** 2)


def rmse(y_hat, batch):
    """Root Mean Squared Error."""
    y = batch[1]
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def mape(y_hat, batch):
    """Mean Absolute Percentage Error (%).  Returns NaN when any |y|=0."""
    y = batch[1]
    return torch.mean(torch.abs((y - y_hat) / (torch.abs(y) + 1e-8))) * 100.0


def smape(y_hat, batch):
    """Symmetric Mean Absolute Percentage Error (%).  Bounded in [0, 200]."""
    y = batch[1]
    denom = torch.abs(y) + torch.abs(y_hat) + 1e-8
    return torch.mean(2.0 * torch.abs(y - y_hat) / denom) * 100.0


def r2(y_hat, batch):
    """Coefficient of determination R².  R²=1 → perfect; R²<0 → worse than mean."""
    y     = batch[1]
    ss_res = torch.sum((y - y_hat) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ── Standalone variants (no batch wrapper) ────────────────────────────────────

def mae_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_hat - y))


def mse_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


def rmse_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_hat - y) ** 2))


def mape_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs((y - y_hat) / (torch.abs(y) + 1e-8))) * 100.0


def smape_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    denom = torch.abs(y) + torch.abs(y_hat) + 1e-8
    return torch.mean(2.0 * torch.abs(y - y_hat) / denom) * 100.0


def r2_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((y - y_hat) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ── Default metric set ────────────────────────────────────────────────────────

def default_ts_metrics():
    return {
        "mae":   mae,
        "mse":   mse,
        "rmse":  rmse,
        "mape":  mape,
        "smape": smape,
        "r2":    r2,
    }
