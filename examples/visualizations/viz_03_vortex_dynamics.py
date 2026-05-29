"""Visualization 03 — Vortex Dynamics: Lamb-Oseen Vortex Pair.

Physics
-------
Two counter-rotating Lamb-Oseen vortices evolving in viscous fluid:

    ω_i(x,y,t) = (Γ_i / (4πν(t+t₀))) · exp(−|r−r_i|² / (4ν(t+t₀)))

    t₀ = R_c² / (4ν)   (viscous core regularisation)

Vortex 1: centre (−d, 0), circulation Γ₁ = +1   (counterclockwise)
Vortex 2: centre (+d, 0), circulation Γ₂ = −1   (clockwise)

This is the analytical solution — no PINN training required for the
vorticity field itself.  A PINN is trained to recover the stream-function
from the vorticity (Poisson: −∇²ψ = ω).

What this shows
---------------
  • Beautiful red-blue vorticity field resembling real PIV measurements.
  • Velocity vector arrows showing the induced flow.
  • How the two vortices diffuse and interact over time.

Run
---
    python -m examples.visualizations.viz_03_vortex_dynamics
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

NU   = 0.01   # kinematic viscosity
D    = 0.6    # half-separation
RC   = 0.15   # vortex core radius  (√(4ν·t₀) = RC  →  t₀ = RC²/4ν)
GAMMA1 =  1.0
GAMMA2 = -1.0
T0   = RC**2 / (4 * NU)


def vorticity(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    t_eff = t + T0
    def vort_i(xc, yc, G):
        r2 = (x - xc)**2 + (y - yc)**2
        return G / (4 * math.pi * NU * t_eff) * np.exp(-r2 / (4 * NU * t_eff))
    return vort_i(-D, 0.0, GAMMA1) + vort_i(+D, 0.0, GAMMA2)


def velocity(x: np.ndarray, y: np.ndarray, t: float) -> tuple:
    t_eff = t + T0
    def vel_i(xc, yc, G):
        dx = x - xc
        dy = y - yc
        r2 = dx**2 + dy**2 + 1e-8
        # Biot-Savart for Oseen vortex
        factor = G / (2 * math.pi * r2) * (1 - np.exp(-r2 / (4 * NU * t_eff)))
        return -factor * dy, factor * dx
    u1, v1 = vel_i(-D, 0.0, GAMMA1)
    u2, v2 = vel_i(+D, 0.0, GAMMA2)
    return u1 + u2, v1 + v2


def visualize() -> None:
    L  = 2.0
    NX = 200
    NA = 20           # arrow grid
    xs = np.linspace(-L, L, NX, dtype=np.float32)
    xg, yg = np.meshgrid(xs, xs)

    t_snapshots = [0.01, 0.3, 0.8, 2.0]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    norm = mcolors.TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)

    for ax, t in zip(axes, t_snapshots):
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)

        omega = vorticity(xg, yg, t)
        im = ax.imshow(omega, origin="lower", extent=[-L, L, -L, L],
                       cmap="RdBu_r", aspect="equal", norm=norm)

        # Velocity arrows
        xa = np.linspace(-L, L, NA)
        xag, yag = np.meshgrid(xa, xa)
        u, v = velocity(xag, yag, t)
        spd = np.sqrt(u**2 + v**2)
        spd = np.where(spd < 1e-6, 1e-6, spd)
        ax.quiver(xag, yag, u / spd, v / spd,
                  color="white", alpha=0.4, scale=30,
                  headwidth=3, headlength=4, width=0.003)

        ax.set_title(f"t = {t:.2f}", color="#e6edf3", fontsize=9)
        ax.set_xlabel("x", color="#8b949e", fontsize=8)
        ax.set_ylabel("y", color="#8b949e", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cb.ax.tick_params(colors="#8b949e", labelsize=8)
    cb.set_label("vorticity ω", color="#8b949e", fontsize=9)

    fig.suptitle(
        "Lamb-Oseen Vortex Pair — Vorticity Evolution  (ν=0.01)",
        color="#e6edf3", fontsize=11, y=1.02)
    plt.tight_layout()
    out = "viz_03_vortex_dynamics.png"
    plt.savefig(out, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved {out}")


def main() -> None:
    print("─" * 55)
    print("  Viz 03 — Vortex Dynamics (Lamb-Oseen pair)")
    print("─" * 55)
    visualize()
    print("  Done (analytical — no training required)")


if __name__ == "__main__":
    main()
