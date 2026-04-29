"""Lesson 08 — Uncertainty Quantification.

Physics problem
---------------
Same 1D heat equation as lesson 03:
    u_t = α u_xx,  α=0.05,  exact: exp(−α π² t) sin(πx)

We train a PINN and then quantify HOW CONFIDENT we are in its predictions.

Three UQ methods (all from pinneaple_uq)
-----------------------------------------
  1. MC Dropout  — cheapest, add dropout + N stochastic forward passes
  2. Deep Ensemble  — most reliable, train M independent networks
  3. Conformal Prediction  — coverage guarantees without architecture change

What you will learn
-------------------
  • Where is the PINN most uncertain? (near boundaries, in sparse regions)
  • How do the prediction intervals compare across methods?
  • What "calibrated" means: a 90% interval should contain 90% of true values

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_08_uq
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle UQ imports
from pinneaple_uq import (
    MCDropoutWrapper, MCDropoutConfig,
    EnsembleUQ,       EnsembleConfig,
    ConformalPredictor,
    CalibrationMetrics,
)

ALPHA  = 0.05
T_MAX  = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 0


# ── Train a single heat PINN ───────────────────────────────────────────────
def train_heat_pinn(dropout_rate: float = 0.0,
                    seed: int = SEED) -> nn.Module:
    torch.manual_seed(seed)
    layers: list[nn.Module] = [nn.Linear(2, 64), nn.Tanh()]
    for _ in range(2):
        layers += [nn.Linear(64, 64), nn.Tanh()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
    layers.append(nn.Linear(64, 1))
    net = nn.Sequential(*layers).to(DEVICE)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5_000)

    for _ in range(5_000):
        opt.zero_grad()
        # Interior
        x  = torch.rand(1500, 1, device=DEVICE)
        t  = torch.rand(1500, 1, device=DEVICE) * T_MAX
        xt = torch.cat([x, t], dim=1).requires_grad_(True)
        u  = net(xt)
        g  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_t = g[:, 1:2]
        u_x = g[:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        l_pde = (u_t - ALPHA * u_xx).pow(2).mean()

        # BCs & IC
        x_bc = torch.cat([torch.zeros(150, 1), torch.ones(150, 1)], dim=0).to(DEVICE)
        t_bc = torch.rand(300, 1, device=DEVICE) * T_MAX
        xt_bc = torch.cat([x_bc, t_bc], dim=1)
        l_bc = net(xt_bc).pow(2).mean()

        x_ic = torch.rand(150, 1, device=DEVICE)
        xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
        u_ic  = torch.sin(math.pi * x_ic)
        l_ic  = (net(xt_ic) - u_ic).pow(2).mean()

        (l_pde + 10.0 * l_bc + 100.0 * l_ic).backward()
        opt.step(); sch.step()

    return net


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 08 — Uncertainty Quantification")
    print("─" * 60)
    print(f"\n  Heat equation: u_t = {ALPHA} u_xx  →  exp(−α π² t) sin(πx)\n")

    # ── Test grid ─────────────────────────────────────────────────────────
    x_np = np.linspace(0, 1, 60, dtype=np.float32)
    t_np = np.linspace(0, T_MAX, 60, dtype=np.float32)
    xg, tg = np.meshgrid(x_np, t_np)
    xt_test = torch.tensor(np.stack([xg.ravel(), tg.ravel()], axis=1), device=DEVICE)
    u_exact = np.exp(-ALPHA * math.pi**2 * tg.ravel()) * np.sin(math.pi * xg.ravel())

    # ── Method 1: MC Dropout ──────────────────────────────────────────────
    print("  [1/3]  MC Dropout (dropout_rate=0.05, N_samples=100)...")
    net_mc = train_heat_pinn(dropout_rate=0.05, seed=SEED)

    mc_cfg  = MCDropoutConfig(n_samples=100, dropout_p=0.05, seed=SEED)
    mc_wrap = MCDropoutWrapper(net_mc, config=mc_cfg)
    uq_mc   = mc_wrap.predict_with_uncertainty(xt_test, device=DEVICE)

    cov_mc  = float(np.mean(
        (u_exact >= (uq_mc.mean.ravel() - 2*uq_mc.std.ravel())) &
        (u_exact <= (uq_mc.mean.ravel() + 2*uq_mc.std.ravel()))
    ))
    print(f"    Coverage of ±2σ interval: {cov_mc*100:.1f}%  (target: ~95%)")

    # ── Method 2: Deep Ensemble ───────────────────────────────────────────
    print("  [2/3]  Deep Ensemble (5 members)...")
    members = [train_heat_pinn(seed=s) for s in range(5)]

    ens_cfg  = EnsembleConfig(n_members=5, seed=SEED)
    ensemble = EnsembleUQ(models=members, config=ens_cfg)
    uq_ens   = ensemble.predict_with_uncertainty(xt_test, device=DEVICE)

    cov_ens = float(np.mean(
        (u_exact >= (uq_ens.mean.ravel() - 2*uq_ens.std.ravel())) &
        (u_exact <= (uq_ens.mean.ravel() + 2*uq_ens.std.ravel()))
    ))
    print(f"    Coverage of ±2σ interval: {cov_ens*100:.1f}%  (target: ~95%)")

    # ── Method 3: Conformal Prediction ───────────────────────────────────
    print("  [3/3]  Conformal Prediction (coverage target = 90%)...")
    net_conf = train_heat_pinn(seed=SEED)

    # Calibration set: 500 random points
    rng    = np.random.default_rng(SEED)
    x_cal  = rng.uniform(0, 1, 500).astype(np.float32)
    t_cal  = rng.uniform(0, T_MAX, 500).astype(np.float32)
    xt_cal = torch.tensor(np.stack([x_cal, t_cal], axis=1), device=DEVICE)
    u_cal  = torch.tensor(
        (np.exp(-ALPHA * math.pi**2 * t_cal) * np.sin(math.pi * x_cal))[:, None],
        device=DEVICE
    )

    cp = ConformalPredictor(
        model = lambda x: net_conf(x),
        alpha = 0.10,    # target 90% coverage
    )
    cp.calibrate(xt_cal, u_cal)

    y_pred, lower, upper = cp.predict(xt_test)
    cov_cp = cp.coverage(xt_test,
                          torch.tensor(u_exact[:, None], dtype=torch.float32, device=DEVICE))
    print(f"    Empirical coverage: {cov_cp*100:.1f}%  (target: 90%)")
    print(f"    Calibrated interval half-width: {cp.quantile:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    def _plot_field(ax, values, title, cmap="plasma"):
        im = ax.imshow(values.reshape(60, 60), origin="lower",
                       extent=[0, 1, 0, T_MAX], cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("t")

    _plot_field(axes[0, 0], uq_mc.mean.ravel(),  "MC Dropout — mean")
    _plot_field(axes[0, 1], uq_mc.std.ravel(),   "MC Dropout — uncertainty (std)", "YlOrRd")
    _plot_field(axes[0, 2], uq_ens.std.ravel(),  "Deep Ensemble — uncertainty (std)", "YlOrRd")

    _plot_field(axes[1, 0], uq_ens.mean.ravel(),  "Deep Ensemble — mean")
    _plot_field(axes[1, 1], (upper - lower).cpu().numpy().ravel(),
                "Conformal — interval width", "YlOrRd")

    ax = axes[1, 2]
    methods = ["MC Dropout\n(±2σ)", "Deep Ensemble\n(±2σ)", "Conformal\n(90%)"]
    coverages = [cov_mc * 100, cov_ens * 100, cov_cp * 100]
    targets   = [95.0,          95.0,           90.0]
    bars = ax.bar(methods, coverages, color=["steelblue", "crimson", "forestgreen"],
                  alpha=0.8, edgecolor="k", lw=0.5)
    for m, tgt, bar in zip(methods, targets, bars):
        ax.axhline(tgt, color="k", ls="--", lw=1, xmin=bar.get_x() - 0.01,
                   xmax=bar.get_x() + bar.get_width() + 0.01)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Coverage vs target (dashed)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Lesson 08 — Uncertainty Quantification  (heat equation)", fontsize=12)
    plt.tight_layout()
    out = "lesson_08_uq.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print(f"""
  Coverage summary (% of true values inside prediction interval):
    MC Dropout  (±2σ)   : {cov_mc*100:.1f}%  (target ≈95%)
    Deep Ensemble (±2σ) : {cov_ens*100:.1f}%  (target ≈95%)
    Conformal (90%)     : {cov_cp*100:.1f}%  (guaranteed ≥90%)

  Key takeaways:
    1. MC Dropout: cheapest — add Dropout to existing net, no retraining.
    2. Deep Ensemble: most reliable — 5 independent nets + variance.
    3. Conformal: the only method with a formal coverage guarantee.
       Requires a calibration set but no architecture changes.
    4. Uncertainty is highest near boundaries and at early times —
       exactly where the PINN has least support from the PDE.

  Next lesson:
    python -m pinneaple_learning.course.lesson_09_time_marching
""")


if __name__ == "__main__":
    main()
