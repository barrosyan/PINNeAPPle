"""Lesson 06 — Model Architectures: SIREN, ModifiedMLP, Fourier Features.

Physics problem
---------------
Helmholtz equation (high-frequency solution):
    u_xx + k² u = 0,    x ∈ [0, 2π/k]
    u(0) = 0,  u'(0) = k  (initial condition for sine)
    Exact: u(x) = sin(kx)

With k = 8, the solution oscillates 8 times — this is hard for a standard
Tanh MLP due to the "spectral bias" (smooth networks resist learning
high-frequency functions).

What you will learn
-------------------
  • Standard Tanh MLP — spectral bias in action
  • SIREN (sinusoidal activations) — designed for high-frequency problems
  • ModifiedMLP (Fourier features + highway gating) — enhanced information flow
  • FourierFeatureEmbedding — embed inputs in Fourier space first

Key classes
-----------
  from pinneaple_models import SIREN, ModifiedMLP, FourierFeatureEmbedding

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_06_model_architectures
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PINNeAPPle model imports for this lesson
from pinneaple_models import SIREN, ModifiedMLP, FourierFeatureEmbedding

K       = 8          # wave number — higher k = harder problem
X_END   = 2*math.pi/K
EPOCHS  = 10_000
N_COL   = 400
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"


# ── Standard Tanh MLP ─────────────────────────────────────────────────────
def make_tanh_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


# ── Fourier MLP: embed x → Fourier features first ─────────────────────────
class FourierMLP(nn.Module):
    def __init__(self, n_fourier: int = 64, sigma: float = 10.0):
        super().__init__()
        self.embed = FourierFeatureEmbedding(in_dim=1, n_fourier=n_fourier,
                                              sigma=sigma, trainable=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_fourier, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(x))


# ── Helper: extract scalar from SIREN/ModifiedMLP ModelOutput ─────────────
def _call(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = net(x)
    if hasattr(out, "y"):      # SIREN and ModifiedMLP return ModelOutput
        return out.y
    return out                 # plain nn.Module returns tensor directly


# ── PDE residual: u_xx + k² u = 0 ────────────────────────────────────────
def helmholtz_residual(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.requires_grad_(True)
    u   = _call(net, x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_xx + K**2 * u


# ── Generic training loop ─────────────────────────────────────────────────
def train_model(net: nn.Module, label: str) -> tuple[list, np.ndarray]:
    net = net.to(DEVICE)
    torch.manual_seed(42)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    x_col = torch.linspace(0, X_END, N_COL, device=DEVICE).unsqueeze(1)
    x0    = torch.zeros(1, 1, device=DEVICE)

    history = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        # PDE residual
        r = helmholtz_residual(net, x_col.clone())
        l_pde = r.pow(2).mean()

        # IC: u(0) = 0
        l_ic_pos = _call(net, x0).pow(2)

        # IC: u'(0) = k
        x0g  = x0.clone().requires_grad_(True)
        u0   = _call(net, x0g)
        u0_x = torch.autograd.grad(u0.sum(), x0g, create_graph=True)[0]
        l_ic_vel = (u0_x - K).pow(2)

        loss = l_pde + 100.0 * (l_ic_pos + l_ic_vel)
        loss.backward()
        opt.step()
        sch.step()
        history.append(float(loss))

    x_np = np.linspace(0, X_END, 500, dtype=np.float32)
    with torch.no_grad():
        u_pred = _call(net, torch.tensor(x_np[:, None], device=DEVICE)).cpu().numpy().ravel()
    return history, u_pred


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("─" * 60)
    print("  Lesson 06 — Model Architectures: SIREN, ModifiedMLP, Fourier")
    print("─" * 60)
    print(f"\n  Helmholtz:  u'' + k²u = 0,  k={K}  (exact: sin({K}x))")
    print(f"  Domain: [0, {X_END:.3f}]  — {K} full oscillations\n")

    models = {
        "Tanh MLP":      make_tanh_mlp(),
        "SIREN":         SIREN(in_dim=1, out_dim=1, hidden_dim=128, n_layers=5,
                                omega_0=30.0),
        "ModifiedMLP":   ModifiedMLP(in_dim=1, out_dim=1, hidden_dim=128, n_layers=6,
                                      n_fourier=32, sigma=K * 1.5),
        "Fourier+MLP":   FourierMLP(n_fourier=64, sigma=K * 1.5),
    }

    results = {}
    x_np   = np.linspace(0, X_END, 500, dtype=np.float32)
    u_exact = np.sin(K * x_np)

    for label, net in models.items():
        print(f"  Training  {label} ...")
        history, u_pred = train_model(net, label)
        rel_l2 = float(
            np.sqrt(((u_pred - u_exact)**2).mean()) / (np.abs(u_exact).mean() + 1e-8)
        )
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        results[label] = {
            "history": history, "pred": u_pred,
            "l2": rel_l2, "params": n_params,
        }
        print(f"    → L2={rel_l2:.3e}  params={n_params:,}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]

    for ax, (label, res), c in zip(axes.ravel(), results.items(), colors):
        ax.plot(x_np, u_exact,  "k-",  lw=2.5, label="Exact  sin(kx)")
        ax.plot(x_np, res["pred"], "--", color=c, lw=1.5,
                label=f"PINN  L2={res['l2']:.1e}  ({res['params']:,} params)")
        ax.set_title(label, fontsize=11, color=c)
        ax.set_xlabel("x"); ax.set_ylabel("u(x)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Lesson 06 — Helmholtz  k={K}  (High-frequency benchmark)", fontsize=12)
    plt.tight_layout()
    out = "lesson_06_model_architectures.png"
    plt.savefig(out, dpi=130)
    print(f"\n  Saved {out}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n  {'Architecture':<18}  {'L2 error':>10}  {'Params':>8}")
    print("  " + "─" * 42)
    for label, res in results.items():
        print(f"  {label:<18}  {res['l2']:>10.3e}  {res['params']:>8,}")

    print("""
  Key takeaways:
    1. Tanh MLP fails on high-frequency problems — spectral bias.
    2. SIREN uses sin(ω₀ Wx) activations and principled init.
       It naturally represents high-frequency functions.
    3. ModifiedMLP uses Fourier features + highway gating.
       More stable but slower to converge than SIREN.
    4. FourierFeatureEmbedding maps x → [cos(Bx), sin(Bx)].
       Simple and effective — just prepend to any MLP.

  Rule of thumb:
    k < 5  →  Tanh MLP is fine
    k ≥ 5  →  Use SIREN or Fourier features
    Very large domain / multi-scale  →  ModifiedMLP (highway gating)

  Next lesson:
    python -m pinneaple_learning.course.lesson_07_inverse_problem
""")


if __name__ == "__main__":
    main()
