import os
import torch
from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_iterable import ZarrUPDIterable
from pinneaple_data.collate import collate_upd_supervised

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics
from pinneaple_train.preprocess import PreprocessPipeline

# ----------------------------
# 1) Build a tiny UPD dataset and write to Zarr
# ----------------------------
out_dir = "examples/_out/end_to_end_01"
os.makedirs(out_dir, exist_ok=True)
zarr_path = os.path.join(out_dir, "toy_ts.zarr")

if not os.path.isdir(zarr_path):
    samples = []
    for i in range(256):
        x = torch.randn(64, 8)   # (T,D)
        y = torch.randn(64, 2)   # (T,2)
        samples.append(
            PhysicalSample(
                state={"x": x, "y": y},
                domain={"type": "grid"},
                provenance={"i": i, "source": "toy"},
                schema={"units": {"x": "arb", "y": "arb"}},
            )
        )
    UPDZarrStore.write(zarr_path, samples, manifest={"name": "toy_ts"})

# ----------------------------
# 2) Stream from Zarr with workers
# ----------------------------
ds = ZarrUPDIterable(zarr_path, fields=["x", "y"], coords=[])
dl = DataLoader(
    ds,
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    collate_fn=collate_upd_supervised,
)

train_loader = dl
val_loader = dl

# ----------------------------
# 3) Model
# ----------------------------
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)

model = M()

# ----------------------------
# 4) Preprocess: FFT feature that preserves T=64
# ----------------------------
class FFTGlobalFeatureStep:
    def __init__(self, reduce="mean", x_key="x"):
        assert reduce in ("mean", "max")
        self.reduce = reduce
        self.x_key = x_key

    def apply(self, batch: dict) -> dict:
        x = batch[self.x_key]

        had_no_batch = (x.dim() == 2)
        if had_no_batch:
            x = x.unsqueeze(0)  # (1,T,D)

        X = torch.fft.rfft(x, dim=-2)  # (B, T//2+1, D)
        mag = X.abs()

        if self.reduce == "mean":
            g = mag.mean(dim=-2)       # (B,D)
        else:
            g = mag.amax(dim=-2)       # (B,D)

        T = x.shape[-2]
        g_bt = g.unsqueeze(-2).expand(-1, T, -1)  # (B,T,D)

        x_aug = torch.cat([x, g_bt], dim=-1)      # (B,T,2D)

        if had_no_batch:
            x_aug = x_aug.squeeze(0)

        batch[self.x_key] = x_aug
        return batch

    def __call__(self, batch: dict) -> dict:
        return self.apply(batch)

preprocess = PreprocessPipeline(
    steps=[FFTGlobalFeatureStep(reduce="mean")]
)

# ----------------------------
# 5) Loss and Trainer
# ----------------------------
combined = CombinedLoss(
    supervised=SupervisedLoss("mse"),
    physics=None,
    w_supervised=1.0,
    w_physics=0.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    metrics=default_metrics(),
    preprocess=preprocess,
)

cfg = TrainConfig(
    epochs=2,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name="demo_ts_fft",
    seed=123,
    deterministic=False,
    save_best=True,
)

out = trainer.fit(train_loader, val_loader, cfg)
print("best_val:", out["best_val"])
print("best_path:", out["best_path"])