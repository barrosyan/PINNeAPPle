"""19_fno_neural_operator.py — Fourier Neural Operator (FNO) for 1D Burgers.

Demonstrates:
- FNO1d architecture from pinneaple_models.neural_operators
- Operator learning: map initial condition u0(x) → solution u(x,T)
- Training on a dataset of (u0, u_T) pairs generated from a simple solver
- Evaluation: relative L2 error over test trajectories
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.neural_operators.fno import FNO1d


# ---------------------------------------------------------------------------
# Data generation: 1D Burgers via simple spectral / Euler solver
# u_t + 0.5 (u²)_x = ν u_xx,  x ∈ [0, 2π] (periodic), T = 1.0
# ---------------------------------------------------------------------------

NU   = 0.01
T    = 1.0
NX   = 64
DT   = 1e-3
N_STEPS = int(T / DT)


def solve_burgers_spectral(u0: np.ndarray) -> np.ndarray:
    """Pseudo-spectral Burgers solver (Fourier, periodic BCs)."""
    u = u0.copy().astype(complex)
    k  = np.fft.fftfreq(NX, d=1.0 / NX)  # wave numbers

    for _ in range(N_STEPS):
        u_hat   = np.fft.fft(u)
        # Diffusion (spectral)
        u_hat  -= DT * NU * (1j * k) ** 2 * u_hat
        u       = np.fft.ifft(u_hat).real
        # Nonlinear (pseudo-spectral)
        u2_hat  = np.fft.fft(0.5 * u ** 2)
        u_hat   = np.fft.fft(u)
        u_hat  -= DT * (1j * k) * u2_hat
        u       = np.fft.ifft(u_hat).real

    return u.real


def generate_dataset(n_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, NX, endpoint=False)
    u0_all, uT_all = [], []
    for _ in range(n_samples):
        # Random IC: sum of low-frequency modes
        n_modes = rng.integers(2, 6)
        u0 = np.zeros(NX)
        for _ in range(n_modes):
            k_mode = rng.integers(1, 4)
            amp    = rng.uniform(-1, 1)
            phase  = rng.uniform(0, 2 * np.pi)
            u0    += amp * np.sin(k_mode * x + phase)
        uT = solve_burgers_spectral(u0)
        u0_all.append(u0.astype(np.float32))
        uT_all.append(uT.astype(np.float32))
    return np.array(u0_all), np.array(uT_all)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset -------------------------------------------------------------
    print("Generating Burgers dataset ...")
    n_train, n_test = 800, 100
    u0_tr, uT_tr = generate_dataset(n_train, seed=0)
    u0_te, uT_te = generate_dataset(n_test,  seed=1)

    # FNO expects (batch, n_x, channels)
    X_tr = torch.tensor(u0_tr[:, :, None], device=device)   # (N, 64, 1)
    Y_tr = torch.tensor(uT_tr[:, :, None], device=device)   # (N, 64, 1)
    X_te = torch.tensor(u0_te[:, :, None], device=device)
    Y_te = torch.tensor(uT_te[:, :, None], device=device)

    # --- FNO -----------------------------------------------------------------
    fno = FNO1d(
        n_modes=16,
        hidden_channels=32,
        n_layers=4,
        in_channels=1,
        out_channels=1,
    ).to(device)

    n_params = sum(p.numel() for p in fno.parameters())
    print(f"FNO1d parameters: {n_params:,}")

    # --- Training ------------------------------------------------------------
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    batch_size = 64
    n_epochs   = 200
    history    = []

    for epoch in range(1, n_epochs + 1):
        fno.train()
        idx = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            bx = X_tr[idx[i: i + batch_size]]
            by = Y_tr[idx[i: i + batch_size]]
            optimizer.zero_grad()
            y_hat = fno(bx)
            loss  = (y_hat - by).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        scheduler.step()
        history.append(epoch_loss / (n_train // batch_size))

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d} | train loss = {history[-1]:.4e}")

    # --- Evaluation ----------------------------------------------------------
    fno.eval()
    with torch.no_grad():
        y_pred_te = fno(X_te).cpu().numpy()[:, :, 0]
    y_true_te = Y_te.cpu().numpy()[:, :, 0]

    rel_l2 = np.sqrt(((y_pred_te - y_true_te) ** 2).sum(axis=1)) / \
             np.sqrt((y_true_te ** 2).sum(axis=1))
    print(f"\nTest relative L2 error: {rel_l2.mean():.4e} ± {rel_l2.std():.4e}")

    # --- Visualisation -------------------------------------------------------
    x = np.linspace(0, 2 * np.pi, NX)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: example prediction
    idx_ex = np.argmin(rel_l2)
    axes[0].plot(x, u0_te[idx_ex], "gray", ls="--", label="IC u₀")
    axes[0].plot(x, y_true_te[idx_ex], "k-",  label="True u(T)")
    axes[0].plot(x, y_pred_te[idx_ex], "r--", label="FNO pred")
    axes[0].set_title(f"Best sample (L2={rel_l2[idx_ex]:.3e})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: worst prediction
    idx_ex2 = np.argmax(rel_l2)
    axes[1].plot(x, u0_te[idx_ex2], "gray", ls="--", label="IC u₀")
    axes[1].plot(x, y_true_te[idx_ex2], "k-",  label="True u(T)")
    axes[1].plot(x, y_pred_te[idx_ex2], "r--", label="FNO pred")
    axes[1].set_title(f"Worst sample (L2={rel_l2[idx_ex2]:.3e})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: training loss + L2 histogram
    axes[2].semilogy(history, label="Train loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE loss")
    axes[2].set_title(f"Training  |  mean rel-L2={rel_l2.mean():.3e}")
    axes[2].grid(True, which="both", alpha=0.3)
    ax2 = axes[2].twinx()
    ax2.hist(rel_l2, bins=20, color="orange", alpha=0.5, label="Test L2")
    ax2.set_ylabel("# samples")

    plt.tight_layout()
    plt.savefig("19_fno_neural_operator_result.png", dpi=120)
    print("Saved 19_fno_neural_operator_result.png")


if __name__ == "__main__":
    main()
