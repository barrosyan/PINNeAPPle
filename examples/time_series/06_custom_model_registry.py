"""Registering a custom time-series model in the global ModelRegistry.

The goal:
  - You can add new forecasters (Transformers, LSTMs, xLSTMs, N-BEATS, etc.)
    without modifying core library code.
  - Once registered, TSModelCatalog can build it by name.

Run:
  python examples/pinneaple_timeseries/06_custom_model_registry.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pinneaple_models.registry import ModelRegistry
from pinneaple_timeseries import TimeSeriesSpec, TSDataModule, TSModelCatalog


# 1) Define and register a model
@ModelRegistry.register(
    name="ts_mlp",
    family="timeseries",
    description="Simple direct MLP forecaster (demo custom registry entry).",
    tags=["timeseries", "forecast", "mlp", "example"],
)
class TSMLP(nn.Module):
    def __init__(self, *, input_len: int, horizon: int, n_features: int = 1, hidden: int = 256):
        super().__init__()
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.n_features = int(n_features)
        self.net = nn.Sequential(
            nn.Flatten(),  # (B, L, F) -> (B, L*F)
            nn.Linear(self.input_len * self.n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.horizon),  # single-target (demo)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, H)


def main() -> None:
    # 2) Create data
    series = torch.randn(3000, 1)
    spec = TimeSeriesSpec(input_len=64, horizon=16, stride=1, target_offset=0)
    train_loader, _ = TSDataModule(series=series, spec=spec, batch_size=64).make_loaders()

    # 3) Build through the catalog by name
    cat = TSModelCatalog()
    print("Catalog models (subset):", [m for m in cat.list() if m.startswith("ts_")][:10])

    model = cat.build("ts_mlp", input_len=spec.input_len, horizon=spec.horizon, n_features=1, hidden=128)
    print("Built model:", model.__class__.__name__)

    # 4) Forward pass smoke test
    x, y = next(iter(train_loader))
    yhat = model(x)
    print("x:", tuple(x.shape), "y:", tuple(y.shape), "yhat:", tuple(yhat.shape))

    print("\nNext step: integrate with pinneaple_train.Trainer + CombinedLoss + BacktestRunner.")


if __name__ == "__main__":
    main()