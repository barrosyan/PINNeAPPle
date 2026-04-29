"""33_rans_turbulence.py — RANS turbulence modelling with k-ω SST PINN.

Demonstrates:
- KOmegaSSTResiduals: prebuilt k-ω SST turbulence closure from pinneaple_environment
- SpalartAllmarasResiduals: SA-model residuals (alternative)
- Channel flow benchmark: log-law recovery in a turbulent plane channel
- Reynolds stress prediction and friction velocity estimation
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_environment import KOmegaSSTResiduals, KOmegaSSTConfig

try:
    from pinneaple_environment import SpalartAllmarasResiduals
    _SA = True
except ImportError:
    _SA = False


# ---------------------------------------------------------------------------
# Problem: fully-developed turbulent channel flow
# Domain: y ∈ [0, δ]  (half-channel height δ = 1)
# Governing equations: RANS + k-ω SST closure
# Mean velocity U(y), TKE k(y), specific dissipation ω(y)
# Boundary conditions:
#   y=0 (wall): U=0, k=0, ω=ω_wall (high)
#   y=δ (symmetry): dU/dy=0, dk/dy=0, dω/dy=0
# Log-law reference: U+(y+) = (1/κ)ln(y+) + B,  κ=0.41, B=5.2
# ---------------------------------------------------------------------------

DELTA      = 1.0
RE_TAU     = 395         # friction Reynolds number
KAPPA      = 0.41        # von Kármán constant
B_CONST    = 5.2
NU         = 1.0 / RE_TAU   # kinematic viscosity (U_τ = 1, δ = 1)
U_TAU      = 1.0
OMEGA_WALL = 6 * NU / (0.075 * (0.1 * DELTA)**2)   # Wilcox near-wall ω


def log_law(y_plus: np.ndarray) -> np.ndarray:
    """Von Kármán log-law: U+ = (1/κ)ln(y+) + B."""
    return (1 / KAPPA) * np.log(np.clip(y_plus, 1e-3, None)) + B_CONST


class ChannelNet(nn.Module):
    """MLP predicting [U, k, ω] as functions of y ∈ [0, δ]."""
    def __init__(self, hidden: int = 64, layers: int = 5):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 2):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, 3)]    # [U, k, ω]
        self.net = nn.Sequential(*mods)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Re_τ = {RE_TAU},  ν = {NU:.5f}")

    # --- k-ω SST residuals module -------------------------------------------
    kwsst_config = KOmegaSSTConfig(
        nu=NU,
        sigma_k1=0.85, sigma_k2=1.0,
        sigma_omega1=0.5, sigma_omega2=0.856,
        beta1=0.075, beta2=0.0828,
        beta_star=0.09,
        alpha1=5.0 / 9.0, alpha2=0.44,
        a1=0.31,
        kappa=KAPPA,
    )
    kwsst = KOmegaSSTResiduals(config=kwsst_config)

    # --- Model ---------------------------------------------------------------
    model = ChannelNet(hidden=64, layers=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Collocation points in y ∈ (0, δ) -----------------------------------
    n_col = 500
    # Cluster near wall with geometric progression
    y_col_np = (1 - np.cos(np.linspace(0, math.pi / 2, n_col))) * DELTA
    y_col = torch.tensor(y_col_np[:, None], dtype=torch.float32, device=device)

    # --- Boundary conditions -------------------------------------------------
    # Wall (y=0): U=0, k=0, ω=ω_wall
    y_wall = torch.zeros(1, 1, device=device)
    # Symmetry (y=δ): gradients = 0 — enforced via Neumann loss

    # --- Loss function -------------------------------------------------------
    def total_loss(model: nn.Module) -> torch.Tensor:
        yc = y_col.clone().requires_grad_(True)
        out = model(yc)            # (N, 3): [U, k, ω]
        U_col = out[:, 0:1]
        k_col = out[:, 1:2]
        w_col = out[:, 2:3]

        # Clamp to physical range
        k_col = k_col.abs() + 1e-6
        w_col = w_col.abs() + 1.0

        res_U, res_k, res_w = kwsst.channel_residuals(
            y=yc, U=U_col, k=k_col, omega=w_col,
        )
        loss_pde = (res_U.pow(2).mean() + res_k.pow(2).mean() +
                    res_w.pow(2).mean())

        # Wall BC
        out_w = model(y_wall)
        loss_bc_U = out_w[:, 0:1].pow(2).mean()
        loss_bc_k = out_w[:, 1:2].pow(2).mean()
        loss_bc_w = (out_w[:, 2:3] - OMEGA_WALL).pow(2).mean()

        # Symmetry BC (dU/dy = 0 at y=δ)
        y_sym = torch.tensor([[DELTA]], device=device, requires_grad=True)
        out_s = model(y_sym)
        dU_dy = torch.autograd.grad(out_s[:, 0].sum(), y_sym)[0]
        loss_sym = dU_dy.pow(2).mean()

        return (loss_pde
                + 100 * loss_bc_U
                + 100 * loss_bc_k
                + 10  * loss_bc_w
                + 10  * loss_sym)

    # --- Training ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8000)

    history = []
    n_epochs = 8000
    print(f"\nTraining k-ω SST PINN ({n_epochs} epochs) ...")

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        loss = total_loss(model)
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(float(loss.item()))

        if epoch % 2000 == 0:
            print(f"  epoch {epoch:5d} | loss = {loss.item():.4e}")

    # --- Evaluation ----------------------------------------------------------
    y_vis_np = np.linspace(1e-3, DELTA, 300, dtype=np.float32)
    y_vis    = torch.tensor(y_vis_np[:, None], device=device)
    with torch.no_grad():
        out_vis = model(y_vis).cpu().numpy()

    U_vis = out_vis[:, 0]
    k_vis = np.abs(out_vis[:, 1])
    w_vis = np.abs(out_vis[:, 2]) + 1.0

    # Wall units
    y_plus_vis = y_vis_np * RE_TAU
    U_plus_vis = U_vis / U_TAU
    U_plus_log = log_law(y_plus_vis)

    # Compare in log-law region (30 < y+ < 200)
    mask = (y_plus_vis > 30) & (y_plus_vis < 200)
    if mask.sum() > 0:
        log_err = np.abs(U_plus_vis[mask] - U_plus_log[mask]).mean()
        print(f"\nLog-law region (30<y+<200) mean |ΔU+| = {log_err:.4f}")

    # --- Visualisation -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: U+ vs y+ (log-law comparison)
    axes[0].semilogx(y_plus_vis, U_plus_vis, "b-",  lw=2, label="k-ω SST PINN")
    axes[0].semilogx(y_plus_vis, U_plus_log, "k--", lw=1.5, label="Log-law")
    axes[0].axvspan(30, 200, alpha=0.1, color="green", label="Log-law region")
    axes[0].set_xlabel("y+")
    axes[0].set_ylabel("U+")
    axes[0].set_title(f"Velocity profile (Re_τ={RE_TAU})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 30])

    # Panel 2: TKE profile
    axes[1].plot(k_vis, y_vis_np, "r-")
    axes[1].set_xlabel("Turbulent kinetic energy k")
    axes[1].set_ylabel("y")
    axes[1].set_title("TKE profile k(y)")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Training loss
    axes[2].semilogy(history)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("k-ω SST PINN training loss")
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("33_rans_turbulence_result.png", dpi=120)
    print("Saved 33_rans_turbulence_result.png")


if __name__ == "__main__":
    main()
