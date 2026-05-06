import os
import torch
import matplotlib.pyplot as plt
from pinneaple_timeseries import TimeSeriesSpec, TSDataModule, TSModelCatalog

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics

series = torch.randn(5000, 8)  # (T,F)

spec = TimeSeriesSpec(input_len=64, horizon=16, stride=1, target_offset=0)
train, val = TSDataModule(series=series, spec=spec, batch_size=64).make_loaders()

catalog = TSModelCatalog()

model = catalog.build_default_fno(
    input_len=spec.input_len,
    horizon=spec.horizon,
    n_features=8,
    n_targets=8,
    width=64,
    modes=16,
    layers=4,
)

combined = CombinedLoss(
    supervised=SupervisedLoss(kind="mse"),
    physics=None,
    w_supervised=1.0,
    w_physics=0.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

os.makedirs("examples/_runs_ts", exist_ok=True)

trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())
cfg = TrainConfig(
    epochs=3, lr=1e-3, device="cpu",
    log_dir="examples/_runs_ts", run_name="fno_ts_demo",
    save_best=True,
)

out = trainer.fit(train, val, cfg)
print(out.get("best_val"), out.get("best_path"))

ckpt = torch.load(out["best_path"], map_location="cpu")
state = ckpt.get("model_state", None)

if state is None:
    state = ckpt.get("state_dict", None)
if state is None:
    state = ckpt

model.load_state_dict(state, strict=True)
model.eval()

x_val, y_val = next(iter(val))

with torch.no_grad():
    y_hat = model(x_val)

for i in range(0,8):
    sample = i
    feature = i

    true = y_val[sample, :, feature].cpu().numpy()
    pred = y_hat[sample, :, feature].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(true, label="Ground Truth")
    plt.plot(pred, label="Prediction")
    plt.title(f"Forecast Comparison (feature={feature})")
    plt.xlabel("Horizon Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()
