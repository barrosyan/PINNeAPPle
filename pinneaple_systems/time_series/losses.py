"""Time series forecast loss functions."""
import torch.nn.functional as F


def mse_forecast(model, y_hat, batch):
    _, y = batch
    loss = F.mse_loss(y_hat, y)
    return {"supervised": loss, "total": loss}
