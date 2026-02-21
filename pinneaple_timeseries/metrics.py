"""Time series metrics (MAE, MSE) for forecasting evaluation."""
import torch


def mae(y_hat, batch):
    y = batch[1]
    return torch.mean(torch.abs(y_hat - y))


def mse(y_hat, batch):
    y = batch[1]
    return torch.mean((y_hat - y) ** 2)


def default_ts_metrics():
    return {"mae": mae, "mse": mse}
