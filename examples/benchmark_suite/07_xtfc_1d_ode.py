"""07 — XtFC: Extreme Theory of Functional Connections for exact BC enforcement.

What this demonstrates
----------------------
- Using XtFC to solve a 1D ODE with *exact* boundary condition satisfaction
- Comparing XtFC (ELM mode) vs VanillaPINN on the same problem
- Using the tfc library for constrained expressions (if installed)
- Ridge regression (one-shot ELM solve) vs gradient descent

Problem
-------
  u'' - u = -2 e^x        x ∈ [0,1]
  u(0) = 0,  u(1) = 0

Analytic solution:
  u(x) = (e^x - e^{x-1} / (e - 1/e)) * ... (see Leake & Mortari 2020)

For this demo we use a simplified linear ODE:
  u'' + u = f(x) = sin(pi*x) * pi^2 - sin(pi*x) = sin(pi*x) * (pi^2 - 1)
  u(0) = 0,  u(1) = 0
  Exact: u(x) = sin(pi*x)

Run from repo root:
    python examples/pinneaple_arena/07_xtfc_1d_ode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from pinneaple_models.pinns.xtfc import XTFC, XTFCConfig, build_xtfc, tfc_available
from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_train import best_device


DEVICE = best_device()
print(f"[Device] {DEVICE}")
print(f"[TFC library] Available: {tfc_available()}")

# ------------------------------------------------------------------
# Problem setup:  u'' + u = f(x), u(0)=u(1)=0, exact: u=sin(πx)
# ------------------------------------------------------------------
f_fn   = lambda x: np.sin(np.pi * x) * (np.pi**2 - 1)   # RHS
u_exact = lambda x: np.sin(np.pi * x)

# Particular solution (for the g + B * N fallback)
def g_fn(x: torch.Tensor) -> torch.Tensor:
    # g(x) = 0  (trivial particular solution since BCs are zero)
    return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

def B_fn(x: torch.Tensor) -> torch.Tensor:
    # B(x) = x * (1 - x) — vanishes at x=0 and x=1
    xi = x[:, 0:1]
    return xi * (1.0 - xi)


# ------------------------------------------------------------------
# 1. XtFC (ELM + TFC constraints)
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("XtFC (Extreme Theory of Functional Connections)")
print("=" * 50)

xtfc = build_xtfc(
    in_dim=1,
    out_dim=1,
    rf_dim=512,
    activation="tanh",
    freeze_random=True,
    g_fn=g_fn,
    B_fn=B_fn,
    use_tfc=True,
    tfc_n=100,
    tfc_deg=20,
    tfc_nC=2,
    tfc_x0=[0.0],
    tfc_xf=[1.0],
)
xtfc = xtfc.to(DEVICE)
print(f"  TFC mode: {xtfc.tfc_ce is not None}")

# Training points
N_TRAIN = 200
x_np = np.linspace(0, 1, N_TRAIN, dtype=np.float32)
f_np = f_fn(x_np).astype(np.float32)

x_t = torch.from_numpy(x_np[:, None]).to(DEVICE)
f_t = torch.from_numpy(f_np[:, None]).to(DEVICE)

# ---- Option A: Ridge regression (one-shot, no gradient descent) ----
# Requires that the constrained expression enforces u''+ u = f  via least-squares
# Here we demonstrate the gradient descent path (Option B) as the universal approach

# ---- Option B: Gradient descent on PDE residual ----
opt_xtfc = torch.optim.Adam(
    [p for p in xtfc.parameters() if p.requires_grad], lr=1e-3
)

EPOCHS = 2000
xtfc_history = []

for epoch in range(EPOCHS):
    opt_xtfc.zero_grad()
    x_req = x_t.requires_grad_(True)
    u     = xtfc.forward_tensor(x_req)   # (N, 1)

    # u''
    du    = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    d2u   = torch.autograd.grad(du, x_req, torch.ones_like(du), create_graph=True)[0]

    res = d2u + u - f_t
    loss = (res ** 2).mean()
    loss.backward()
    opt_xtfc.step()
    xtfc_history.append(float(loss.item()))

# Evaluate
xtfc.eval()
with torch.no_grad():
    u_xtfc = xtfc.forward_tensor(x_t).cpu().numpy().ravel()

u_ref = u_exact(x_np)
err_xtfc = np.abs(u_xtfc - u_ref)
print(f"  XtFC  | max error: {err_xtfc.max():.4e}  |  L2 error: {np.linalg.norm(err_xtfc)/np.linalg.norm(u_ref):.4e}")


# ------------------------------------------------------------------
# 2. Vanilla PINN (for comparison)
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("Vanilla PINN (for comparison)")
print("=" * 50)

vpinn = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 1),
).to(DEVICE)
opt_vp = torch.optim.Adam(vpinn.parameters(), lr=1e-3)

vpinn_history = []
x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32).to(DEVICE)
u_bc = torch.zeros(2, 1, dtype=torch.float32).to(DEVICE)

for epoch in range(EPOCHS):
    opt_vp.zero_grad()
    x_req = x_t.clone().requires_grad_(True)
    u = vpinn(x_req)

    du   = torch.autograd.grad(u, x_req, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    d2u  = torch.autograd.grad(du, x_req, torch.ones_like(du), create_graph=True)[0]

    loss_pde = ((d2u + u - f_t) ** 2).mean()
    loss_bc  = ((vpinn(x_bc) - u_bc) ** 2).mean()
    loss = loss_pde + 100.0 * loss_bc
    loss.backward()
    opt_vp.step()
    vpinn_history.append(float(loss_pde.item()))

with torch.no_grad():
    u_vpinn = vpinn(x_t).cpu().numpy().ravel()

err_vp = np.abs(u_vpinn - u_ref)
print(f"  VPINN | max error: {err_vp.max():.4e}  |  L2 error: {np.linalg.norm(err_vp)/np.linalg.norm(u_ref):.4e}")


# ------------------------------------------------------------------
# 3. Comparison
# ------------------------------------------------------------------
print("\n[Summary]")
print(f"  {'Method':<12} | {'Max Error':>12} | {'Rel L2':>12}")
print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}")
for name, err, ref in [
    ("XtFC",  err_xtfc, u_ref),
    ("VanillaPINN",  err_vp, u_ref),
]:
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(ref)
    print(f"  {name:<12} | {err.max():>12.4e} | {rel_l2:>12.4e}")


# ------------------------------------------------------------------
# 4. Visualization
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "xtfc_1d_ode"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].semilogy(xtfc_history,  label="XtFC (PDE only)", lw=1.5)
    axes[0].semilogy(vpinn_history, label="VanillaPINN (PDE)", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("PDE residual")
    axes[0].set_title("Training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Solutions
    axes[1].plot(x_np, u_ref,   "k-",  lw=2,   label="Exact")
    axes[1].plot(x_np, u_xtfc,  "r--", lw=1.5, label="XtFC")
    axes[1].plot(x_np, u_vpinn, "b:",  lw=1.5, label="VanillaPINN")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title("Solution comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Point-wise errors
    axes[2].semilogy(x_np, err_xtfc, "r-",  lw=1.5, label="XtFC")
    axes[2].semilogy(x_np, err_vp,   "b--", lw=1.5, label="VanillaPINN")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("|error|")
    axes[2].set_title("Point-wise absolute error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "xtfc_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\n[Plot] {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available.")


print("\n=== COMPLETE ===")
