"""pinneaple_train example 05: DistributedDataParallel (DDP) with torchrun.

This example is intentionally small and focuses on *how to launch* DDP.

Requirements
------------
- PyTorch built with distributed support.
- For GPU DDP: NCCL available.

Run (CPU / gloo)
---------------
torchrun --standalone --nproc_per_node=2 examples/pinneaple_train/05_ddp_torchrun.py

Run (GPU / nccl)
---------------
torchrun --standalone --nproc_per_node=2 examples/pinneaple_train/05_ddp_torchrun.py --device cuda --backend nccl

Notes
-----
- Only rank0 writes logs/checkpoints.
- The Trainer automatically wraps loaders with DistributedSampler.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_train import Trainer, TrainConfig, CombinedLoss, SupervisedLoss


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device base name")
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"], help="DDP backend")
    args = ap.parse_args()

    os.makedirs("examples/_runs", exist_ok=True)

    # toy data
    x = torch.randn(4096, 8)
    y = torch.randn(4096, 2)
    train = DataLoader(TensorDataset(x[:3200], y[:3200]), batch_size=128, shuffle=True)
    val = DataLoader(TensorDataset(x[3200:], y[3200:]), batch_size=256)

    model = M()
    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None, w_supervised=1.0, w_physics=0.0)

    def loss_fn(m, y_hat, batch):
        return combined(m, y_hat, batch)

    trainer = Trainer(model=model, loss_fn=loss_fn)

    # IMPORTANT: set ddp=True. Under torchrun, RANK/WORLD_SIZE/LOCAL_RANK are injected.
    cfg = TrainConfig(
        epochs=3,
        lr=1e-3,
        device=args.device,
        ddp=True,
        ddp_backend=args.backend,
        log_dir="examples/_runs",
        run_name="ddp_demo",
        seed=123,
        deterministic=False,
        amp=False,
        save_best=True,
    )

    out = trainer.fit(train, val, cfg)
    if out.get("rank0"):
        print("DDP rank0 finished. best_path:", out.get("best_path"))


if __name__ == "__main__":
    main()