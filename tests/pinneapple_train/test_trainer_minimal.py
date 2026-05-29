import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneapple_train.trainer import Trainer, TrainConfig
from pinneapple_train.losses import CombinedLoss, SupervisedLoss
from pinneapple_train.metrics import default_metrics


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.net(x)


def test_trainer_runs_one_epoch(tmp_path):
    x = torch.randn(64, 4)
    y = torch.randn(64, 2)
    dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    model = Tiny()

    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None)

    def loss_fn(model_, y_hat, batch):
        return combined(model_, y_hat, batch)

    tr = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())

    cfg = TrainConfig(
        epochs=1,
        lr=1e-3,
        device="cpu",
        log_dir=str(tmp_path),
        run_name="t",
        deterministic=False,
        save_best=False,
    )

    out = tr.fit(dl, dl, cfg)
    assert "history" in out
    assert len(out["history"]) == 1
