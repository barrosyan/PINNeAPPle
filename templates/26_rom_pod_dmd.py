"""26_rom_pod_dmd.py — Reduced Order Models: POD and DMD.

Demonstrates:
- PODReducedModel: Proper Orthogonal Decomposition via SVD
- DMDModel: Dynamic Mode Decomposition for linear system identification
- HAVOKModel: Hankel Alternative View of Koopman (delay embedding)
- ROM reconstruction error and future-state prediction
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_models.rom.pod import PODReducedModel
from pinneaple_models.rom.dmd import DMDModel

try:
    from pinneaple_models.rom.havok import HAVOKModel
    _HAVOK = True
except ImportError:
    _HAVOK = False


# ---------------------------------------------------------------------------
# Synthetic dataset: 2D wave equation snapshots
# u(x,t) = Σₙ aₙ(t) φₙ(x)   where  aₙ(t) = cos(ωₙ t + ψₙ)
# This produces a low-rank spatiotemporal field (ideal for POD/DMD).
# ---------------------------------------------------------------------------

NX       = 100
N_MODES  = 4
N_SNAPS  = 200
DT_SIM   = 0.05
N_PRED   = 40     # future steps to predict with DMD


def generate_snapshots(nx: int = NX, n_modes: int = N_MODES,
                       n_snaps: int = N_SNAPS) -> tuple[np.ndarray, np.ndarray]:
    """Return spatial modes ψ (nx,) and snapshots X (nx, n_snaps)."""
    x = np.linspace(0, 2 * np.pi, nx)
    t = np.arange(n_snaps) * DT_SIM
    X = np.zeros((nx, n_snaps), dtype=np.float32)
    for n in range(1, n_modes + 1):
        omega = 0.5 * n
        psi_n = np.sin(n * x / 2)
        amp   = 1.0 / n
        phase = n * 0.3
        X    += amp * np.outer(psi_n, np.cos(omega * t + phase))
    return x, X.astype(np.float32)


def main():
    np.random.seed(0)
    print("Generating snapshot matrix ...")
    x_grid, X = generate_snapshots()    # X: (nx, n_snaps)

    # Add small noise
    X_noisy = X + np.random.normal(0, 0.02, X.shape).astype(np.float32)

    # Split: train on first 160, predict next 40
    n_train = N_SNAPS - N_PRED
    X_train = X_noisy[:, :n_train]
    X_test  = X[:, n_train:]             # clean ground truth for evaluation

    # =========================================================================
    # 1) POD reconstruction
    # =========================================================================
    print("\n[1] POD reconstruction ...")
    for n_comp in [2, 4, 8]:
        pod = PODReducedModel(n_components=n_comp)
        pod.fit(X_train)
        X_recon = pod.reconstruct(X_train)
        rel_err = np.sqrt(((X_recon - X_train)**2).sum()) / \
                  np.sqrt((X_train**2).sum())
        ev_ratio = pod.explained_variance_ratio()
        print(f"  r={n_comp:2d}  recon error={rel_err:.4e}  "
              f"explained variance={ev_ratio:.4f}")

    # Full POD for visualisation (r=4)
    pod4 = PODReducedModel(n_components=4)
    pod4.fit(X_train)
    X_pod_recon = pod4.reconstruct(X_train)

    # =========================================================================
    # 2) DMD for future prediction
    # =========================================================================
    print("\n[2] DMD forecasting ...")
    dmd = DMDModel(n_modes=10, dt=DT_SIM)
    dmd.fit(X_train)

    # Predict future states
    X_dmd_pred = dmd.predict(n_steps=N_PRED, x0=X_train[:, -1])
    dmd_err = np.sqrt(((X_dmd_pred - X_test)**2).mean())
    print(f"  DMD forecast RMSE = {dmd_err:.4e}")

    # =========================================================================
    # 3) HAVOK (Hankel-based Koopman, if available)
    # =========================================================================
    if _HAVOK:
        print("\n[3] HAVOK (Hankel-DMD) ...")
        # Use a single sensor time series
        sensor_signal = X_train[NX // 2, :]    # single sensor at x = π
        havok = HAVOKModel(n_delay=20, n_modes=5, dt=DT_SIM)
        havok.fit(sensor_signal)
        signal_pred = havok.predict(n_steps=N_PRED)
        truth_sensor = X_test[NX // 2, :]
        h_err = np.sqrt(((signal_pred[:len(truth_sensor)] - truth_sensor)**2).mean())
        print(f"  HAVOK forecast RMSE = {h_err:.4e}")

    # =========================================================================
    # Visualisation
    # =========================================================================
    t_full = np.arange(N_SNAPS) * DT_SIM
    t_train = t_full[:n_train]
    t_test  = t_full[n_train:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Snapshot matrix (space-time heatmap)
    im = axes[0, 0].imshow(X, aspect="auto", extent=[0, t_full[-1], 0, 2 * np.pi],
                            origin="lower", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    plt.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title("Ground truth snapshot matrix u(x,t)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("x")

    # Panel 2: POD reconstruction at midpoint
    axes[0, 1].plot(x_grid, X_train[:, 80],      "k-",  label="Truth")
    axes[0, 1].plot(x_grid, X_pod_recon[:, 80],  "r--", label="POD (r=4) recon")
    axes[0, 1].set_title("POD snapshot reconstruction (t=4.0 s)")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: DMD prediction for a single x
    ix = NX // 3
    axes[1, 0].plot(t_train, X_train[ix, :],  "b-",  label="Train")
    axes[1, 0].plot(t_test,  X_test[ix, :],   "k--", label="Truth (test)")
    axes[1, 0].plot(t_test,  X_dmd_pred[ix, :], "r-", label="DMD forecast")
    axes[1, 0].set_title(f"DMD forecast at x={x_grid[ix]:.2f}")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: POD singular values
    sv = pod4.singular_values()
    axes[1, 1].bar(range(1, len(sv) + 1), sv, color="steelblue")
    axes[1, 1].set_title("POD singular values")
    axes[1, 1].set_xlabel("Mode index")
    axes[1, 1].set_ylabel("Singular value")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("26_rom_pod_dmd_result.png", dpi=120)
    print("\nSaved 26_rom_pod_dmd_result.png")


if __name__ == "__main__":
    main()
