"""16_uncertainty_quantification.py — Uncertainty Quantification for PINNs.

Demonstrates:
- MCDropoutEstimator for epistemic uncertainty via Monte-Carlo Dropout
- DeepEnsemble for variance estimation across independently trained models
- ConformalPredictor for distribution-free prediction intervals
- Uncertainty-aware loss plotting
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_uq.mc_dropout import MCDropoutEstimator
from pinneaple_uq.ensemble import DeepEnsemble
from pinneaple_uq.conformal import ConformalPredictor


# ---------------------------------------------------------------------------
# Problem: 1D regression with physics prior  u'' = -π² sin(πx), x∈[0,1]
# Exact solution: u(x) = sin(πx)
# Noisy observations used to train the ensemble and calibrate conformal
# ---------------------------------------------------------------------------

import math

N_TRAIN = 30
NOISE_STD = 0.05


def u_exact(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


def make_noisy_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n).astype(np.float32)
    y = u_exact(x) + rng.normal(0, NOISE_STD, n).astype(np.float32)
    return x[:, None], y[:, None]


def build_mlp(dropout_p: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Dropout(dropout_p),
        nn.Linear(64, 1),
    )


def train_model(model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor,
                n_epochs: int = 3000, lr: float = 1e-3) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = (model(x_tr) - y_tr).pow(2).mean()
        loss.backward()
        optimizer.step()
    return model


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Training data -------------------------------------------------------
    x_np, y_np = make_noisy_data(N_TRAIN, seed=0)
    x_tr = torch.tensor(x_np, device=device)
    y_tr = torch.tensor(y_np, device=device)

    # Calibration / validation set
    x_cal_np, y_cal_np = make_noisy_data(200, seed=1)

    # Test grid
    x_test_np = np.linspace(0, 1, 200, dtype=np.float32)[:, None]
    x_test = torch.tensor(x_test_np, device=device)

    # =========================================================================
    # 1) MC-Dropout estimator
    # =========================================================================
    print("\n[1] MC-Dropout UQ ...")
    base_model = build_mlp(dropout_p=0.15).to(device)
    train_model(base_model, x_tr, y_tr, n_epochs=3000)

    mc_estimator = MCDropoutEstimator(model=base_model, n_samples=200)
    mc_mean, mc_std = mc_estimator.predict(x_test)        # both (N,) numpy arrays

    # =========================================================================
    # 2) Deep Ensemble
    # =========================================================================
    print("[2] Deep Ensemble UQ (5 members) ...")

    ensemble_members = []
    for i in range(5):
        m = build_mlp(dropout_p=0.0).to(device)
        train_model(m, x_tr, y_tr, n_epochs=3000)
        ensemble_members.append(m)

    deep_ens = DeepEnsemble(models=ensemble_members)
    ens_mean, ens_std = deep_ens.predict(x_test)          # both (N,) numpy arrays

    # =========================================================================
    # 3) Conformal Prediction
    # =========================================================================
    print("[3] Conformal prediction (coverage = 90%) ...")
    # Use the first ensemble member as the base predictor
    cal_preds = ensemble_members[0](
        torch.tensor(x_cal_np, device=device)
    ).detach().cpu().numpy().ravel()
    cal_targets = y_cal_np.ravel()

    conformal = ConformalPredictor(
        model=ensemble_members[0],
        alpha=0.10,                # 90 % coverage
    )
    conformal.calibrate(
        x_cal=torch.tensor(x_cal_np, device=device),
        y_cal=torch.tensor(y_cal_np, device=device),
    )
    conf_lower, conf_upper = conformal.predict_interval(x_test)  # (N,) each

    # =========================================================================
    # Visualisation
    # =========================================================================
    x_plot = x_test_np.ravel()
    u_true = u_exact(x_plot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: MC Dropout
    axes[0].plot(x_plot, u_true, "k--", label="Exact")
    axes[0].plot(x_plot, mc_mean, "b-", label="MC mean")
    axes[0].fill_between(x_plot,
                         mc_mean - 2 * mc_std,
                         mc_mean + 2 * mc_std,
                         alpha=0.3, color="blue", label="±2σ")
    axes[0].scatter(x_np.ravel(), y_np.ravel(), s=15, c="k", zorder=5, label="Obs")
    axes[0].set_title("MC-Dropout UQ")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Deep Ensemble
    axes[1].plot(x_plot, u_true, "k--", label="Exact")
    axes[1].plot(x_plot, ens_mean, "g-", label="Ensemble mean")
    axes[1].fill_between(x_plot,
                         ens_mean - 2 * ens_std,
                         ens_mean + 2 * ens_std,
                         alpha=0.3, color="green", label="±2σ")
    axes[1].scatter(x_np.ravel(), y_np.ravel(), s=15, c="k", zorder=5, label="Obs")
    axes[1].set_title("Deep Ensemble UQ (5 members)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Conformal
    axes[2].plot(x_plot, u_true, "k--", label="Exact")
    axes[2].fill_between(x_plot, conf_lower, conf_upper,
                         alpha=0.4, color="orange", label="90% conformal PI")
    axes[2].scatter(x_np.ravel(), y_np.ravel(), s=15, c="k", zorder=5, label="Obs")
    axes[2].set_title("Conformal Prediction Intervals (90%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("16_uncertainty_quantification_result.png", dpi=120)
    print("Saved 16_uncertainty_quantification_result.png")


if __name__ == "__main__":
    main()
