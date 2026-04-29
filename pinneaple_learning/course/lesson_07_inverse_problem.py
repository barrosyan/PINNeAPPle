"""Lesson 07 — Inverse Problem: Recovering Physical Parameters.

Physics problem
---------------
Heat equation with UNKNOWN diffusivity:
    u_t = α u_xx,    (x,t) ∈ [0,1] × [0,1]
    u(x,0) = sin(πx),   u(0,t)=u(1,t)=0
    Exact: u(x,t) = exp(−α π² t) sin(πx),  α_true = 0.05

Inverse problem:
    Given N=50 noisy observations of u at scattered (x,t) locations,
    recover α_true.

Two approaches
--------------
  1. Gradient-based: α as a trainable nn.Parameter — Adam optimises net
     weights AND α simultaneously.
  2. EKI (Ensemble Kalman Inversion): derivative-free, robust to noise.

Key PINNeAPPle classes used
----------------------------
  from pinneaple_inverse import (InverseProblemSolver, InverseSolverConfig,
                                   EnsembleKalmanInversion, EKIConfig,
                                   GaussianMisfit, PointObsOperator)

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_07_inverse_problem
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle imports
from pinneaple_inverse import (
    InverseProblemSolver, InverseSolverConfig,
    EnsembleKalmanInversion, EKIConfig,
    GaussianMisfit, PointObsOperator,
)

ALPHA_TRUE = 0.05
N_OBS      = 50       # number of noisy observations
NOISE_STD  = 0.01     # observation noise
SEED       = 42
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Network ────────────────────────────────────────────────────────────────
def make_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


# ── Exact solution (used to generate observations) ────────────────────────
def exact(x: np.ndarray, t: np.ndarray, alpha: float = ALPHA_TRUE) -> np.ndarray:
    return np.exp(-alpha * math.pi**2 * t) * np.sin(math.pi * x)


# ── Generate synthetic observations ───────────────────────────────────────
def make_observations(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x_obs = rng.uniform(0.05, 0.95, n).astype(np.float32)
    t_obs = rng.uniform(0.05, 0.95, n).astype(np.float32)
    u_obs = exact(x_obs, t_obs) + rng.normal(0, NOISE_STD, n).astype(np.float32)
    return x_obs, t_obs, u_obs


# ── Approach 1: Gradient-based (alpha as nn.Parameter) ───────────────────
def _heat_residual(net: nn.Module, alpha: torch.Tensor,
                   xt: torch.Tensor) -> torch.Tensor:
    xt  = xt.requires_grad_(True)
    u   = net(xt)
    g1  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
    u_t = g1[:, 1:2]
    u_x = g1[:, 0:1]
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
    return u_t - alpha * u_xx


def train_gradient_based(x_obs, t_obs, u_obs) -> tuple[float, list]:
    torch.manual_seed(SEED)
    net   = make_net().to(DEVICE)
    alpha = nn.Parameter(torch.tensor([0.2], device=DEVICE))  # wrong initial guess

    opt = torch.optim.Adam(list(net.parameters()) + [alpha], lr=1e-3)

    # Collocation points
    x_col = torch.rand(1000, 1, device=DEVICE)
    t_col = torch.rand(1000, 1, device=DEVICE)
    xt_col = torch.cat([x_col, t_col], dim=1)

    # IC points
    x_ic  = torch.rand(200, 1, device=DEVICE)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    u_ic  = torch.sin(math.pi * x_ic)

    # Observations as tensors
    xt_o  = torch.tensor(np.stack([x_obs, t_obs], axis=1), device=DEVICE)
    u_o   = torch.tensor(u_obs[:, None], device=DEVICE)

    alpha_history = []
    for epoch in range(5_000):
        opt.zero_grad()
        r     = _heat_residual(net, alpha, xt_col.clone())
        l_pde = r.pow(2).mean()
        l_ic  = (net(xt_ic) - u_ic).pow(2).mean()
        l_data = (net(xt_o) - u_o).pow(2).mean()
        loss  = l_pde + 100.0 * l_ic + 1000.0 * l_data
        loss.backward()
        opt.step()
        alpha.data.clamp_(1e-4, 1.0)  # keep physically meaningful
        alpha_history.append(float(alpha))

    return float(alpha), alpha_history


# ── Approach 2: EKI (Ensemble Kalman Inversion) ───────────────────────────
def train_eki(x_obs, t_obs, u_obs) -> tuple[float, object]:
    """EKI is derivative-free — pass a forward model as a plain Python function."""
    torch.manual_seed(SEED)
    net = make_net().to(DEVICE)

    def _pretrain(alpha_val: float) -> nn.Module:
        """Quick pretrain the network for a given alpha."""
        m = make_net().to(DEVICE)
        o = torch.optim.Adam(m.parameters(), lr=1e-3)
        x_c = torch.rand(500, 1, device=DEVICE)
        t_c = torch.rand(500, 1, device=DEVICE)
        xt  = torch.cat([x_c, t_c], dim=1)
        x_i = torch.rand(100, 1, device=DEVICE)
        xt_i = torch.cat([x_i, torch.zeros_like(x_i)], dim=1)
        u_i  = torch.sin(math.pi * x_i)
        a    = torch.tensor([alpha_val], device=DEVICE)
        for _ in range(3_000):
            o.zero_grad()
            r    = _heat_residual(m, a, xt.clone())
            l    = r.pow(2).mean() + 100.0 * (m(xt_i) - u_i).pow(2).mean()
            l.backward(); o.step()
        return m

    def forward_model(theta: np.ndarray) -> np.ndarray:
        """theta = [log_alpha].  Returns predicted u at observation locations."""
        alpha_val = float(np.exp(theta[0]))
        m = _pretrain(alpha_val)
        xt_o = torch.tensor(np.stack([x_obs, t_obs], axis=1), device=DEVICE)
        with torch.no_grad():
            u_pred = m(xt_o).cpu().numpy().ravel()
        return u_pred

    eki_cfg = EKIConfig(
        n_ensemble  = 20,
        n_iterations = 15,
        noise_std   = NOISE_STD,
        init_spread = 0.5,
        seed        = SEED,
        verbose     = False,
    )
    eki = EnsembleKalmanInversion(forward_fn=forward_model, config=eki_cfg)

    # Initial ensemble: log(alpha) ~ N(log(0.2), 0.5²)
    theta_init = np.log(0.2) + np.random.default_rng(SEED).normal(0, 0.5, 20)

    print("    Running EKI iterations (each iteration = N_ensemble forward models)...")
    history = eki.solve(y_obs=u_obs, theta_init=theta_init)
    alpha_eki = float(np.exp(history.theta_mean_history[-1][0]))
    return alpha_eki, history


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 07 — Inverse Problem: Recovering Diffusivity")
    print("─" * 60)
    print(f"\n  True α = {ALPHA_TRUE},  noise std = {NOISE_STD},  N_obs = {N_OBS}\n")

    x_obs, t_obs, u_obs = make_observations(N_OBS)

    # ── Approach 1: Gradient-based ────────────────────────────────────────
    print("  [1/2]  Gradient-based recovery (α as nn.Parameter)...")
    alpha_grad, alpha_hist = train_gradient_based(x_obs, t_obs, u_obs)
    print(f"    → Recovered α = {alpha_grad:.5f}  (true = {ALPHA_TRUE})")

    # ── Approach 2: EKI ───────────────────────────────────────────────────
    print("  [2/2]  EKI (Ensemble Kalman Inversion)...")
    alpha_eki, eki_hist = train_eki(x_obs, t_obs, u_obs)
    print(f"    → Recovered α = {alpha_eki:.5f}  (true = {ALPHA_TRUE})")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gradient-based convergence
    ax = axes[0]
    ax.plot(alpha_hist, "steelblue", lw=1.5, label="α estimate")
    ax.axhline(ALPHA_TRUE, color="k", ls="--", lw=2, label=f"True α={ALPHA_TRUE}")
    ax.set_title("Gradient-based: α convergence")
    ax.set_xlabel("Epoch"); ax.set_ylabel("α")
    ax.legend(); ax.grid(True, alpha=0.3)

    # EKI convergence
    ax = axes[1]
    eki_alphas = [np.exp(h[0]) for h in eki_hist.theta_mean_history]
    ax.plot(eki_alphas, "crimson", lw=2, marker="o", ms=5, label="EKI α estimate")
    ax.axhline(ALPHA_TRUE, color="k", ls="--", lw=2, label=f"True α={ALPHA_TRUE}")
    ax.set_title("EKI: α convergence across iterations")
    ax.set_xlabel("EKI iteration"); ax.set_ylabel("α")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Observations
    ax = axes[2]
    ax.scatter(x_obs, t_obs, c=u_obs, cmap="plasma", s=40, zorder=5)
    x_g = np.linspace(0, 1, 60, dtype=np.float32)
    t_g = np.linspace(0, 1, 60, dtype=np.float32)
    xg, tg = np.meshgrid(x_g, t_g)
    u_bg = exact(xg.ravel(), tg.ravel()).reshape(60, 60)
    im = ax.contourf(xg, tg, u_bg, levels=20, cmap="plasma", alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Observations (N={N_OBS}) on exact solution")
    ax.set_xlabel("x"); ax.set_ylabel("t")

    plt.suptitle(f"Lesson 07 — Inverse Problem  (true α={ALPHA_TRUE})", fontsize=12)
    plt.tight_layout()
    out = "lesson_07_inverse_problem.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    print(f"""
  Results summary:
    True α           = {ALPHA_TRUE:.5f}
    Gradient-based   = {alpha_grad:.5f}  (error: {abs(alpha_grad - ALPHA_TRUE)/ALPHA_TRUE*100:.1f}%)
    EKI              = {alpha_eki:.5f}  (error: {abs(alpha_eki - ALPHA_TRUE)/ALPHA_TRUE*100:.1f}%)

  Key takeaways:
    1. Gradient-based: α is just an nn.Parameter — no extra code.
       Works when the forward model is differentiable and smooth.
    2. EKI is derivative-free: you only need a forward model callable.
       Handles discontinuities, non-smooth objectives, multi-modal posteriors.
    3. Both approaches simultaneously optimise the network AND recover α.
    4. More observations and less noise → better recovery.

  Next lesson:
    python -m pinneaple_learning.course.lesson_08_uq
""")


if __name__ == "__main__":
    main()
