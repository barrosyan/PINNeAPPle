"""29_koopman_autoencoder.py — Koopman Autoencoder for nonlinear dynamics.

Demonstrates:
- KoopmanAutoencoder: encoder φ, Koopman operator K, decoder ψ
- Training: reconstruction loss + K-step prediction loss + eigenvalue regularisation
- LinearisedPrediction: multi-step rollout in the Koopman embedding space
- Visualisation of learned Koopman eigenvalues on the unit circle
"""

import cmath
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.autoencoders.koopman_ae import KoopmanAutoencoder
from pinneaple_models.autoencoders.koopman_ae import LinearisedPrediction


# ---------------------------------------------------------------------------
# Synthetic nonlinear system: 2D Duffing oscillator
# dx/dt = y
# dy/dt = -δy - αx - βx³ + γcos(ωt)
# Parameters chosen to give rich quasi-periodic dynamics
# ---------------------------------------------------------------------------

DELTA = 0.2
ALPHA = -1.0
BETA  = 1.0
GAMMA = 0.3
OMEGA = 1.2
DT    = 0.05


def duffing_rk4(state: np.ndarray, t: float) -> np.ndarray:
    x, y = state
    def f(s, tt):
        xx, yy = s
        dxdt = yy
        dydt = -DELTA * yy - ALPHA * xx - BETA * xx**3 + GAMMA * np.cos(OMEGA * tt)
        return np.array([dxdt, dydt])
    k1 = f(state, t)
    k2 = f(state + DT / 2 * k1, t + DT / 2)
    k3 = f(state + DT / 2 * k2, t + DT / 2)
    k4 = f(state + DT * k3, t + DT)
    return state + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_duffing(n_steps: int, x0: np.ndarray, seed: int = 0) -> np.ndarray:
    traj = [x0.copy()]
    state = x0.copy()
    for i in range(n_steps):
        state = duffing_rk4(state, i * DT)
        traj.append(state.copy())
    return np.array(traj, dtype=np.float32)   # (n_steps+1, 2)


def generate_dataset(n_traj: int = 50, n_steps: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        x0 = rng.uniform(-1.5, 1.5, 2)
        trajs.append(simulate_duffing(n_steps, x0))
    return np.stack(trajs)   # (n_traj, n_steps+1, 2)


def main():
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset -------------------------------------------------------------
    print("Simulating Duffing oscillator trajectories ...")
    data = generate_dataset(n_traj=100, n_steps=300)   # (100, 301, 2)
    n_tr = 80
    data_tr = torch.tensor(data[:n_tr], device=device)  # (80, 301, 2)
    data_te = torch.tensor(data[n_tr:], device=device)

    K_STEPS = 5   # multi-step prediction in training loss

    # --- Model ---------------------------------------------------------------
    model = KoopmanAutoencoder(
        state_dim=2,
        latent_dim=16,
        encoder_layers=[32, 32],
        decoder_layers=[32, 32],
        n_koopman_modes=16,
        use_complex_koopman=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Koopman AE parameters: {n_params:,}")

    # --- Training ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    history = {"recon": [], "pred": [], "eig": []}
    n_epochs = 300
    bs       = 16

    for epoch in range(1, n_epochs + 1):
        model.train()
        idx = torch.randperm(n_tr, device=device)
        epoch_r = epoch_p = epoch_e = 0.0
        n_batches = 0

        for i in range(0, n_tr, bs):
            traj = data_tr[idx[i: i + bs]]   # (bs, T, 2)
            x0   = traj[:, :-K_STEPS, :]     # (bs, T-K, 2)
            xt   = traj[:, K_STEPS:, :]      # (bs, T-K, 2) - K-step targets

            optimizer.zero_grad()

            # Reconstruction loss
            x0_flat = x0.reshape(-1, 2)
            z_flat  = model.encode(x0_flat)
            x_recon = model.decode(z_flat)
            l_recon = (x_recon - x0_flat).pow(2).mean()

            # Multi-step prediction loss
            z_pred = z_flat
            l_pred = 0.0
            for _ in range(K_STEPS):
                z_pred = model.koopman_step(z_pred)
            x_pred = model.decode(z_pred)
            l_pred = (x_pred - xt.reshape(-1, 2)).pow(2).mean()

            # Eigenvalue regularisation (keep eigenvalues near unit circle)
            eigs = model.koopman_eigenvalues()   # complex tensor (n_modes,)
            l_eig = (eigs.abs() - 1.0).pow(2).mean()

            loss = l_recon + 0.5 * l_pred + 0.01 * l_eig
            loss.backward()
            optimizer.step()

            epoch_r += float(l_recon.item())
            epoch_p += float(l_pred.item())
            epoch_e += float(l_eig.item())
            n_batches += 1

        scheduler.step()
        history["recon"].append(epoch_r / n_batches)
        history["pred"].append(epoch_p / n_batches)
        history["eig"].append(epoch_e / n_batches)

        if epoch % 60 == 0:
            print(f"  epoch {epoch:3d} | recon={history['recon'][-1]:.3e}  "
                  f"pred={history['pred'][-1]:.3e}  eig={history['eig'][-1]:.3e}")

    # --- Rollout evaluation --------------------------------------------------
    model.eval()
    traj_test  = data_te[0].cpu().numpy()   # (301, 2) ground truth
    x0_t       = data_te[0:1, 0, :]        # (1, 2)

    lin_pred   = LinearisedPrediction(model=model, n_steps=100)
    with torch.no_grad():
        x_rollout = lin_pred.predict(x0_t).cpu().numpy()   # (1, 101, 2)

    x_rollout = x_rollout[0]   # (101, 2)

    rollout_rmse = float(np.sqrt(((x_rollout - traj_test[:101])**2).mean()))
    print(f"\nRollout RMSE (100 steps): {rollout_rmse:.4e}")

    # --- Eigenvalues on unit circle -----------------------------------------
    with torch.no_grad():
        eigs = model.koopman_eigenvalues().cpu().numpy()   # (n_modes,) complex

    # =========================================================================
    # Visualisation
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Phase portrait comparison
    axes[0].plot(traj_test[:101, 0], traj_test[:101, 1], "k-",  lw=1.5, label="True")
    axes[0].plot(x_rollout[:, 0],   x_rollout[:, 1],    "r--", lw=1.5, label="Koopman rollout")
    axes[0].set_title("Duffing phase portrait (100-step rollout)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ẋ")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Training loss
    axes[1].semilogy(history["recon"], label="Reconstruction")
    axes[1].semilogy(history["pred"],  label="K-step prediction")
    axes[1].semilogy(history["eig"],   label="Eigenvalue reg.")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Koopman AE training losses")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    # Panel 3: Koopman eigenvalues on unit circle
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[2].plot(np.cos(theta), np.sin(theta), "gray", lw=1, label="Unit circle")
    axes[2].scatter(eigs.real, eigs.imag, s=50, c="blue", zorder=5, label="Eigenvalues")
    axes[2].set_aspect("equal")
    axes[2].set_title("Koopman eigenvalues")
    axes[2].set_xlabel("Re(λ)")
    axes[2].set_ylabel("Im(λ)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("29_koopman_autoencoder_result.png", dpi=120)
    print("Saved 29_koopman_autoencoder_result.png")


if __name__ == "__main__":
    main()
