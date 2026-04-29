"""Tier 1 Explorer — interactive concept demos.

All functions print rich, formatted explanations to stdout.
No external dependencies beyond the standard library.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quickstart() -> None:
    """Print the Tier 1 quickstart guide."""
    print(_QUICKSTART)


def what_is_pinn() -> None:
    """Explain what a PINN is, with a toy example."""
    print(_WHAT_IS_PINN)


def what_is_loss() -> None:
    """Break down the PINN loss function term by term."""
    print(_WHAT_IS_LOSS)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

_QUICKSTART = """
╔══════════════════════════════════════════════════════════════╗
║          Welcome to Physics AI — Tier 1: Explorer           ║
╚══════════════════════════════════════════════════════════════╝

You are here because you know some physics and want to understand
what machine learning can do with it. Good news: you are already
most of the way there.

─── What is Physics AI? ──────────────────────────────────────

Classical solvers (finite element, finite difference, Runge-Kutta)
solve a specific problem for a specific set of parameters.

Physics AI trains a neural network that APPROXIMATES the solution
function u(x, t; θ) — once trained, it can:
  • Evaluate anywhere (no grid required)
  • Solve the inverse problem (recover unknown parameters)
  • Handle noisy or sparse observations
  • Generalise across parameter families (neural operators)

─── The three tools you will use ─────────────────────────────

  1. PINNeAPPle  — your laboratory for experimenting with all
                   Physics AI approaches. Low boilerplate,
                   lots of examples, easy benchmarking.

  2. PyTorch     — the deep learning engine. PINNeAPPle sits
                   on top of it.

  3. NVIDIA PhysicsNeMo — the production platform. When your
                   approach is validated and you need scale,
                   speed, and enterprise tooling, you migrate.

─── Your first 30 minutes ────────────────────────────────────

  Step 1 — run a simple PINN:

      cd examples/getting_started
      python 01_harmonic_oscillator.py

  Step 2 — open 01_harmonic_oscillator.py and find:
      • Where the network is defined  (build_net)
      • Where the PDE residual is computed  (train → res)
      • Where the initial condition is enforced  (loss_ic)

  Step 3 — change ω (omega) and re-run. Watch the period change.

  Step 4 — read the explanation:

      import pinneaple_learning.tier1_explorer as t1
      t1.what_is_pinn()
      t1.what_is_loss()

─── Full guide ───────────────────────────────────────────────

  pinneaple_learning/tier1_explorer/README.md
"""

_WHAT_IS_PINN = """
╔══════════════════════════════════════════════════════════════╗
║               What is a PINN?                               ║
╚══════════════════════════════════════════════════════════════╝

PINN = Physics-Informed Neural Network

─── The core idea ────────────────────────────────────────────

A neural network u_θ(x, t) is trained to satisfy:

  1. The PDE:       F[u_θ] = 0    everywhere in the domain
  2. The BCs:       u_θ = g       on the boundary
  3. The ICs:       u_θ = h       at t = 0

"Satisfy" means: minimise the squared residuals.

─── Concrete example — harmonic oscillator ───────────────────

PDE:    x''(t) + ω² x(t) = 0
IC:     x(0) = 1,   x'(0) = 0

Network:  x_θ(t)  — maps time t → displacement x

Loss terms:
  L_pde = mean( (x_θ''(t) + ω² x_θ(t))² )   ← PyTorch autograd
  L_ic  = (x_θ(0) - 1)² + (x_θ'(0) - 0)²

Total:   L = L_pde + λ * L_ic

After training, x_θ(t) ≈ cos(ωt).

─── Why does this work? ──────────────────────────────────────

Neural networks are universal function approximators.
By penalising physics violations as a loss, backpropagation
finds weights that make the network behave like the solution.

─── The derivatives are automatic ───────────────────────────

  import torch
  tc = t.clone().requires_grad_(True)
  x  = net(tc)                                   # forward pass
  xd = torch.autograd.grad(x.sum(), tc,
                            create_graph=True)[0] # x'
  xdd = torch.autograd.grad(xd.sum(), tc,
                             create_graph=True)[0] # x''

No finite differences. No stencils. Exact to machine precision.

─── Where PINNs excel ────────────────────────────────────────

  ✓ Irregular domains (no meshing needed)
  ✓ Inverse problems (unknown coefficients in the PDE)
  ✓ Noisy or sparse observations (data + physics hybrid)
  ✓ Parametric families — train once, query many parameters

─── Where PINNs struggle ─────────────────────────────────────

  ✗ High-frequency oscillations (spectral bias)
  ✗ Strongly chaotic systems (Lorenz at long times)
  ✗ Very large 3D domains (use FNO or DeepONet instead)
  ✗ Sharp discontinuities / shocks (need special treatment)

─── Next concept ─────────────────────────────────────────────

  t1.what_is_loss()    ← break down the loss function
"""

_WHAT_IS_LOSS = """
╔══════════════════════════════════════════════════════════════╗
║           The PINN Loss Function — term by term             ║
╚══════════════════════════════════════════════════════════════╝

  L_total = L_pde + λ_bc * L_bc + λ_ic * L_ic + λ_data * L_data

─── L_pde  (PDE residual loss) ──────────────────────────────

  Collocation points:  t_col ~ Uniform(0, T)  or Latin-Hypercube
  Residual at each point:  r_i = F[u_θ](t_col_i)

  L_pde = (1/N) Σ r_i²

  This is the "physics" part. N = 200–2000 collocation points
  is usually enough for 1D / 2D problems.

─── L_bc  (boundary condition loss) ─────────────────────────

  Dirichlet BC  u = g  on ∂Ω:
      L_bc = mean( (u_θ(x_bc) − g(x_bc))² )

  Neumann BC  ∂u/∂n = h  on ∂Ω:
      L_bc = mean( (∂u_θ/∂n(x_bc) − h(x_bc))² )

  Trick: for Dirichlet BCs you can sometimes HARD-ENFORCE them
  by construction:
      u_θ(x) = g(x) + x*(1−x) * net(x)   [1D unit domain]
  This removes L_bc entirely and usually improves training.

─── L_ic  (initial condition loss) ──────────────────────────

  Position IC:   L_ic_pos = (u_θ(t=0) − u_0)²
  Velocity IC:   L_ic_vel = (u_θ'(t=0) − v_0)²

─── L_data  (observation loss) ──────────────────────────────

  When you have measurement data (x_d, y_d):
      L_data = mean( (u_θ(x_d) − y_d)² )

  This is optional. PINNs can train without any data (pure
  physics), but data always helps when available.

─── Loss weights λ ──────────────────────────────────────────

  The weights balance competing objectives. Common defaults:

      λ_ic = 10 – 200      (ICs need more emphasis early)
      λ_bc = 1 – 10
      λ_data = 1 – 100     (depends on data quantity/quality)

  PINNeAPPle's LossBalancer implements self-adaptive weighting
  (residual-based, NTK-based, or manual).

─── Reading the loss curves ──────────────────────────────────

  Healthy training:
      - All three terms decrease together
      - L_pde dominates early, then plateaus

  Red flags:
      - L_ic / L_bc stays high → increase λ
      - L_pde oscillates → reduce learning rate
      - All losses plateau above 1e-4 → add more collocation pts

─── Relative L2 error ────────────────────────────────────────

  Used to compare PINN accuracy against exact / reference:

      ε = ‖u_θ − u_ref‖₂ / ‖u_ref‖₂

  Tier 1 examples typically achieve ε ~ 1e-3 to 1e-2.
  Production-grade PINNs typically achieve ε ~ 1e-4 to 1e-5.
"""
