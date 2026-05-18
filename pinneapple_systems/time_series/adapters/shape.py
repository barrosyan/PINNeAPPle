"""Shape adapters for time series models (e.g. LastStepToHorizon)."""
import torch.nn as nn


class LastStepToHorizon(nn.Module):
    def __init__(self, model, horizon):
        super().__init__()
        self.model = model
        self.horizon = horizon

    def forward(self, x):
        y = self.model(x)
        return y[:, None, :].repeat(1, self.horizon, 1)
