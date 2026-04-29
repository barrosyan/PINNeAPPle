"""Visualization 04 — Allen-Cahn Phase Field Equation.

Physics
-------
Allen-Cahn phase separation in 2D:

    φ_t = ε²(φ_xx + φ_yy) + φ − φ³    in [0,1]²

    W(φ) = ¼(φ²−1)²   (double-well potential, minima at φ=±1)
    ε = 0.05           (interface width parameter)

    Periodic boundary conditions.

Starting from a random noisy initial condition, phases separate into
domains of φ≈+1 (red) and φ≈−1 (blue) with a thin diffuse interface.

Implementation
--------------
The ODE in time is integrated using the pseudospectral method (exact
Laplacian via FFT), so this is an accurate reference for what a
time-marching PINN should learn.

What this shows
---------------
  • Random initial noise → large-scale phase domains (pattern formation).
  • The interface sharpens and coarsens over time.
  • Topological changes: connected regions form and merge.

Run
---
    python -m examples.visualizations.viz_04_phase_field
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


EPS   = 0.04
N     = 256
DT    = 1e-3
T_MAX = 0.5


def run_allen_cahn(n: int = N, dt: float = DT, t_max: float = T_MAX,
                   seed: int = 0) -> list[tuple[float, np.ndarray]]:
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-0.1, 0.1, (n, n))

    # Wavenumber arrays for pseudospectral approach
    k  = np.fft.fftfreq(n, d=1.0/n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2

    # Integrating factor for the stiff linear term
    # ∂φ/∂t = ε²Δφ + (φ − φ³)
    # Linear: ε²Δ  →  implicit (Crank-Nicolson or exact exponential)
    # Nonlinear: φ − φ³  →  explicit
    eps2_k2 = EPS**2 * k2

    snapshots: list[tuple[float, np.ndarray]] = [(0.0, phi.copy())]
    t = 0.0
    save_times = {0.1, 0.25, 0.5}

    while t < t_max - 1e-10:
        phi_hat = np.fft.fft2(phi)

        # Nonlinear part
        nl = phi - phi**3

        # Semi-implicit step: (1 − dt·ε²·k²)φ_hat_new = φ_hat + dt·fft(nl)
        phi_hat_new = (phi_hat + dt * np.fft.fft2(nl)) / (1.0 + dt * eps2_k2)
        phi = np.real(np.fft.ifft2(phi_hat_new))
        phi = np.clip(phi, -1.0, 1.0)
        t  += dt

        for st in list(save_times):
            if abs(t - st) < dt * 0.6:
                snapshots.append((t, phi.copy()))
                save_times.discard(st)

    return snapshots


def visualize(snapshots: list[tuple[float, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 4.5),
                             facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    cmap = "RdBu_r"
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    for ax, (t, phi) in zip(axes, snapshots):
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)

        im = ax.imshow(phi, origin="lower", extent=[0, 1, 0, 1],
                       cmap=cmap, norm=norm, aspect="equal",
                       interpolation="bilinear")
        # Mark zero-crossing (interface)
        ax.contour(phi, levels=[0.0], colors=["#3fb950"],
                   linewidths=1.0, extent=[0, 1, 0, 1])

        ax.set_title(f"t = {t:.2f}", color="#e6edf3", fontsize=9)
        ax.set_xlabel("x", color="#8b949e", fontsize=8)
        ax.set_ylabel("y", color="#8b949e", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=25, pad=0.02)
    cb.ax.tick_params(colors="#8b949e", labelsize=8)
    cb.set_label("φ  (phase field)", color="#8b949e", fontsize=9)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["−1 (blue phase)", "interface", "+1 (red phase)"])

    fig.suptitle(
        "Allen-Cahn Phase Separation  φ_t = ε²Δφ + φ − φ³  ·  ε=0.04",
        color="#e6edf3", fontsize=11, y=1.02)
    plt.tight_layout()
    out = "viz_04_phase_field.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 04 — Allen-Cahn Phase Field")
    print("─" * 55)
    print("  Running pseudospectral integration...")
    snapshots = run_allen_cahn()
    visualize(snapshots)


if __name__ == "__main__":
    main()
