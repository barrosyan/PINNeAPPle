import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)

# data
x = torch.randn(1024, 8)
y = torch.randn(1024, 2)

train = DataLoader(TensorDataset(x[:800], y[:800]), batch_size=64, shuffle=True)
val = DataLoader(TensorDataset(x[800:], y[800:]), batch_size=128)

# CombinedLoss matches Trainer signature via wrapper below
combined = CombinedLoss(
    supervised=SupervisedLoss(kind="mse"),
    physics=None,
    w_supervised=1.0,
    w_physics=0.0,
)

def loss_fn(model, y_hat, batch):
    # CombinedLoss already returns {"supervised", "total", ...}
    return combined(model, y_hat, batch)

os.makedirs("examples/_runs", exist_ok=True)

trainer = Trainer(model=M(), loss_fn=loss_fn, metrics=default_metrics())

cfg = TrainConfig(
    epochs=3,
    lr=1e-3,
    device="cpu",
    log_dir="examples/_runs",
    run_name="demo",
    seed=123,
    deterministic=False,
    amp=False,
    save_best=True,
)

out = trainer.fit(train, val, cfg)
print("best_val:", out.get("best_val"))
print("best_path:", out.get("best_path"))
