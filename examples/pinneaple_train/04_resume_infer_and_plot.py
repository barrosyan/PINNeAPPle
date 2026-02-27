"""pinneaple_train example 04: Resume / inference / plotting.

This script demonstrates the *full* training lifecycle:

1) Train a tiny model (fast) with audit logs + best checkpoint
2) Load the best checkpoint with pinneaple_train.inference.load_checkpoint
3) Run inference with pinneaple_train.inference.predict
4) Plot training curves with pinneaple_train.viz.plot_history

Run
---
python examples/pinneaple_train/04_resume_infer_and_plot.py

Notes
-----
- matplotlib is used for plotting (a window may pop up depending on your environment).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_train import Trainer, TrainConfig, CombinedLoss, SupervisedLoss
from pinneaple_train.inference import load_checkpoint, predict
from pinneaple_train.viz import plot_history


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main() -> None:
    os.makedirs("examples/_runs", exist_ok=True)

    # 1) Small supervised regression: y = sin(x)
    g = torch.Generator().manual_seed(0)
    x = (2 * math.pi) * (torch.rand(2048, 1, generator=g) - 0.5)
    y = torch.sin(x)

    train = DataLoader(TensorDataset(x[:1600], y[:1600]), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(x[1600:], y[1600:]), batch_size=256)

    model = TinyMLP()

    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None, w_supervised=1.0, w_physics=0.0)

    def loss_fn(m, y_hat, batch):
        return combined(m, y_hat, batch)

    trainer = Trainer(model=model, loss_fn=loss_fn)

    cfg = TrainConfig(
        epochs=10,
        lr=2e-3,
        device="cpu",
        log_dir="examples/_runs",
        run_name="resume_demo",
        seed=7,
        deterministic=False,
        amp=False,
        save_best=True,
    )

    out = trainer.fit(train, val, cfg)
    best_path = out.get("best_path")
    print("best_path:", best_path)

    # 2) Resume / load
    model2 = TinyMLP()
    if best_path is None:
        raise RuntimeError("No best_path was produced. Set TrainConfig.save_best=True.")
    ckpt = load_checkpoint(model2, best_path, map_location="cpu")
    print("loaded keys:", list(ckpt.keys()))

    # 3) Inference
    xs = torch.linspace(-math.pi, math.pi, 200).unsqueeze(1)
    yh = predict(model2, xs, device="cpu")
    mse = torch.mean((yh - torch.sin(xs)) ** 2).item()
    print("MSE on dense grid:", mse)

    # 4) Plot history
    jsonl_path = os.path.join(cfg.log_dir, f"{cfg.run_name}.jsonl")
    hist = read_jsonl(jsonl_path)

    # plot_history expects a list of dicts with epoch/train_total/val_total
    plot_history(hist, keys=("train_total", "val_total"))


if __name__ == "__main__":
    main()