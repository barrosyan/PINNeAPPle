"""Lesson 11 — Operator Learning with AFNO.

The key difference from PINNs
-------------------------------
A PINN solves ONE problem: given α=0.05, solve u_t = 0.05 u_xx.
An operator learns the MAP: α ↦ u(·,·). Train once, query any α.

Problem: parametric heat equation
-----------------------------------
    u_t = α u_xx,    (x,t) ∈ [0,1]²
    u(x,0) = sin(πx),  u(0,t)=u(1,t)=0
    Exact: u(x,t) = exp(−α π² t) sin(πx)
    α ∈ [0.01, 0.5]  (the parameter we learn to generalise over)

Operator approach
-----------------
Input:  (x, t, α)          — query coordinates + parameter
Output: u(x, t; α)         — solution at those coordinates

This is simpler than FNO (which acts on functions).
We use PINNeAPPle's AFNO architecture — Adaptive Fourier Neural Operator —
as the backbone, augmented with the parameter embedding.

For a full parametric operator (FNO-style), see templates/19_fno_neural_operator.py.

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_11_operator_learning
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
from pinneaple_models import SIREN, FourierFeatureEmbedding
from pinneaple_validate import compare_to_analytical

ALPHA_TRAIN = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]   # training parameters
ALPHA_TEST  = [0.03, 0.07, 0.15, 0.35]              # interpolation test
N_PER_PARAM = 1_500                                  # collocation pts per param
EPOCHS      = 10_000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── Parametric network: inputs are (x, t, α) ─────────────────────────────
class ParametricHeatNet(nn.Module):
    """Maps (x, t, α) → u. Three inputs: spatial coords + parameter."""

    def __init__(self):
        super().__init__()
        # Fourier features for (x, t) — handles high-frequency spatial structure
        self.embed_xt    = FourierFeatureEmbedding(in_dim=2, n_fourier=32, sigma=3.0)
        # Parameter encoder: α → 16-dim embedding
        self.param_enc   = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
        )
        in_dim = 2 * 32 + 32  # fourier(xt) + param_enc(α)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        xt_feat    = self.embed_xt(torch.cat([x, t], dim=1))
        alpha_feat = self.param_enc(alpha)
        return self.trunk(torch.cat([xt_feat, alpha_feat], dim=1))


# ── Exact solution ─────────────────────────────────────────────────────────
def exact_fn(x: torch.Tensor, t: torch.Tensor,
             alpha: float) -> torch.Tensor:
    return torch.exp(-alpha * math.pi**2 * t) * torch.sin(math.pi * x)


# ── Training ───────────────────────────────────────────────────────────────
def train_parametric_operator() -> ParametricHeatNet:
    torch.manual_seed(0)
    net = ParametricHeatNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    def sample_batch():
        """Sample (x, t, α) collocation points across all training parameters."""
        x_list, t_list, a_list = [], [], []
        for alpha_val in ALPHA_TRAIN:
            n = N_PER_PARAM
            x  = torch.rand(n, 1)
            t  = torch.rand(n, 1)
            a  = torch.full((n, 1), alpha_val)
            x_list.append(x); t_list.append(t); a_list.append(a)
        return (torch.cat(x_list, 0).to(DEVICE),
                torch.cat(t_list, 0).to(DEVICE),
                torch.cat(a_list, 0).to(DEVICE))

    for epoch in range(EPOCHS):
        opt.zero_grad()
        x, t, a = sample_batch()

        # ── PDE residual for each (x, t, α) ──────────────────────────────
        xt = torch.cat([x, t], dim=1).requires_grad_(True)
        x_ = xt[:, 0:1]; t_ = xt[:, 1:2]
        u  = net(x_, t_, a)
        g  = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_t  = g[:, 1:2]
        u_x  = g[:, 0:1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        l_pde = (u_t - a * u_xx).pow(2).mean()

        # ── BCs: u(0,t)=0 and u(1,t)=0 ──────────────────────────────────
        n_bc = 200 * len(ALPHA_TRAIN)
        t_bc = torch.rand(n_bc, 1, device=DEVICE)
        a_bc = torch.tensor(
            np.repeat(ALPHA_TRAIN, 200), dtype=torch.float32, device=DEVICE
        ).unsqueeze(1)
        l_bc = (net(torch.zeros(n_bc, 1, device=DEVICE), t_bc, a_bc).pow(2).mean() +
                net(torch.ones(n_bc,  1, device=DEVICE), t_bc, a_bc).pow(2).mean())

        # ── IC: u(x,0) = sin(πx) ─────────────────────────────────────────
        n_ic = 200 * len(ALPHA_TRAIN)
        x_ic = torch.rand(n_ic, 1, device=DEVICE)
        a_ic = torch.tensor(
            np.repeat(ALPHA_TRAIN, 200), dtype=torch.float32, device=DEVICE
        ).unsqueeze(1)
        u_ic = torch.sin(math.pi * x_ic)
        l_ic = (net(x_ic, torch.zeros_like(x_ic), a_ic) - u_ic).pow(2).mean()

        loss = l_pde + 10.0 * l_bc + 100.0 * l_ic
        loss.backward()
        opt.step(); sch.step()

        if epoch % 2000 == 0:
            print(f"    epoch {epoch:5d}  loss={float(loss):.4e}  "
                  f"pde={float(l_pde):.4e}")

    return net


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 11 — Operator Learning (Parametric Heat Equation)")
    print("─" * 60)
    print(f"\n  Train on α ∈ {ALPHA_TRAIN}")
    print(f"  Test on  α ∈ {ALPHA_TEST}  (never seen during training)\n")

    print("  Training parametric operator (x, t, α) → u...")
    net = train_parametric_operator()

    # ── Evaluate: training α and test α ───────────────────────────────────
    print("\n  Evaluating on training and test parameters:")
    all_alphas  = ALPHA_TRAIN + ALPHA_TEST
    is_train    = [True] * len(ALPHA_TRAIN) + [False] * len(ALPHA_TEST)

    x_np  = np.linspace(0, 1, 50, dtype=np.float32)
    t_np  = np.linspace(0, 1, 50, dtype=np.float32)
    xg, tg = np.meshgrid(x_np, t_np)
    xy_flat = np.stack([xg.ravel(), tg.ravel()], axis=1)

    results = {}
    for alpha_val, train_flag in zip(all_alphas, is_train):
        x_t  = torch.tensor(xg.ravel()[:, None], device=DEVICE)
        t_t  = torch.tensor(tg.ravel()[:, None], device=DEVICE)
        a_t  = torch.full_like(x_t, alpha_val)

        with torch.no_grad():
            u_pred = net(x_t, t_t, a_t).cpu().numpy().ravel()
        u_ex  = np.exp(-alpha_val * math.pi**2 * tg.ravel()) * np.sin(math.pi * xg.ravel())
        l2    = float(np.sqrt(((u_pred - u_ex)**2).mean()) /
                      (np.abs(u_ex).mean() + 1e-8))
        tag   = "train" if train_flag else "TEST"
        results[alpha_val] = (u_pred, l2, tag)
        print(f"    α={alpha_val:.3f}  [{tag:5s}]  L2={l2:.4e}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    all_alphas_sorted = sorted(results.keys())

    for ax, alpha_val in zip(axes.ravel(), all_alphas_sorted[:10]):
        u_pred, l2, tag = results[alpha_val]
        u_ex = np.exp(-alpha_val * math.pi**2 * tg.ravel()) * np.sin(math.pi * xg.ravel())
        err  = np.abs(u_pred - u_ex).reshape(50, 50)
        im   = ax.imshow(err, origin="lower", extent=[0,1,0,1],
                         cmap="hot", aspect="auto", vmin=0)
        color = "green" if tag == "train" else "crimson"
        ax.set_title(f"α={alpha_val:.3f} [{tag}]\nL2={l2:.1e}",
                     fontsize=8, color=color)
        ax.set_xlabel("x"); ax.set_ylabel("t")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle("Lesson 11 — Operator Learning: |Error| maps\n"
                 "Green=training α, Red=test α (interpolation)", fontsize=11)
    plt.tight_layout()
    out = "lesson_11_operator_learning.png"
    plt.savefig(out, dpi=120)
    print(f"\n  Saved {out}")

    print("""
  Key takeaways:
    1. One parametric network trained on {α₁,...,αₙ} generalises to new α
       WITHOUT re-training — this is the operator learning paradigm.
    2. The parameter encoder embeds α into a learned feature space.
       FourierFeatureEmbedding handles the spatial/temporal frequencies.
    3. Interpolation (test α between training α values) works well.
       Extrapolation (test α far outside training range) is risky.
    4. For very large parametric families (100+ parameters), use FNO or
       DeepONet — see templates/19_fno_neural_operator.py.

  Next (final) lesson:
    python -m pinneaple_learning.course.lesson_12_to_physicsnemo
""")


if __name__ == "__main__":
    main()
