"""
pinneaple_solvers -> feature engineering -> pinneaple_train Trainer

Goal:
  - compute a solver-derived feature (FFT magnitude) from x
  - append feature to x
  - train a tiny supervised model

This example is defensive: if FFTSolver signature differs, adapt in one place.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics


# ---- solver import (adjust if your module path differs)
from pinneaple_solvers.fft import FFTSolver
from pinneaple_train.preprocess import PreprocessPipeline, SolverFeatureStep


class M(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    out_dir = "examples/_out/solvers_fft_feature_train"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Toy data: x is (B, D). We'll pretend D samples are a signal.
    # ------------------------------------------------------------
    x = torch.randn(2048, 64)  # signal length = 64
    y = torch.randn(2048, 2)

    train = DataLoader(TensorDataset(x[:1600], y[:1600]), batch_size=64, shuffle=True)
    val = DataLoader(TensorDataset(x[1600:], y[1600:]), batch_size=128, shuffle=False)

    # ------------------------------------------------------------
    # 2) Solver + preprocess
    # ------------------------------------------------------------
    fft_solver = FFTSolver()

    preprocess = PreprocessPipeline(
        steps=[
            SolverFeatureStep(
                solver=fft_solver,
                mode="append",          # append feature to x
                select_var_dim=None,
                reduce_fft_to="magnitude",
            )
        ]
    )

    # We don't know feature dim a priori, so probe one batch after preprocess.fit/apply.
    # We'll do it manually here to set model input dim.
    peek_batch = next(iter(train))
    peek = {"x": peek_batch[0], "y": peek_batch[1]}

    preprocess.fit([peek])  # MVP fit
    peek2 = preprocess.apply(peek)
    x2 = peek2["x"]
    in_dim = x2.shape[-1]
    out_dim = y.shape[-1]

    model = M(in_dim=in_dim, out_dim=out_dim)

    # ------------------------------------------------------------
    # 3) Loss + trainer
    # ------------------------------------------------------------
    combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None)

    def loss_fn(model, y_hat, batch):
        # y_hat is already tensor here
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
        run_name="fft_feature_demo",
        seed=123,
        deterministic=False,
        save_best=True,
    )

    out = trainer.fit(train, val, cfg)
    print("best_val:", out["best_val"])
    print("best_path:", out["best_path"])


if __name__ == "__main__":
    main()
