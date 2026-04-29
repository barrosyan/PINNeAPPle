# Tier 1 — Explorer: What is Physics AI?

> **Who this is for:** You understand physics (ODEs, PDEs, conservation laws) but are new to Physics AI / PINNs.

---

## Table of contents

1. [What is Physics AI?](#1-what-is-physics-ai)
2. [The Physics AI pipeline](#2-the-physics-ai-pipeline)
3. [What is a PINN?](#3-what-is-a-pinn)
4. [Understanding the loss function](#4-understanding-the-loss-function)
5. [Running your first example](#5-running-your-first-example)
6. [What to experiment with](#6-what-to-experiment-with)
7. [Where PINNs struggle — and what comes next](#7-where-pinns-struggle)
8. [Milestones before advancing to Tier 2](#8-milestones)

---

## 1. What is Physics AI?

Classical numerical solvers (FEM, FDM, Runge-Kutta) compute the solution of a PDE for one specific set of parameters on a fixed grid. They are **exact but expensive** to re-run.

Physics AI trains a **neural network** to approximate the solution function $u(x, t; \theta)$ directly. Once trained:

| Capability | Classical solver | Physics AI |
|-----------|-----------------|-----------|
| Single-point evaluation | Re-run the full solver | Sub-millisecond forward pass |
| New parameters | Full re-solve | Fine-tune or operator inference |
| Inverse problem | Adjoint solvers (complex) | Direct gradient descent |
| Irregular domains | Meshing required | No grid needed |
| Noisy observations | Hard to integrate | Built-in loss term |

Physics AI does **not replace** classical solvers. It is a complementary tool that excels in specific regimes.

---

## 2. The Physics AI pipeline

```
Physics problem
      │
      ▼
Define domain + BCs/ICs
      │
      ▼
Choose architecture          ← PINN, FNO, DeepONet, GNN …
      │
      ▼
Write PDE residual as loss   ← automatic differentiation
      │
      ▼
Train (minimize total loss)
      │
      ▼
Validate (L2 error, BC check, conservation check)
      │
      ▼
Deploy / Scale
      │
      ▼
(Production) NVIDIA PhysicsNeMo
```

PINNeAPPle covers every step in this pipeline. PhysicsNeMo provides the production runtime for the final step.

---

## 3. What is a PINN?

A **Physics-Informed Neural Network** encodes the physics directly into the training objective.

**Example — harmonic oscillator:**

$$x''(t) + \omega^2 x(t) = 0, \quad x(0) = 1, \quad x'(0) = 0$$

The network $x_\theta(t)$ is trained by minimizing:

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{i=1}^{N}\left(x_\theta''(t_i) + \omega^2 x_\theta(t_i)\right)^2}_{\mathcal{L}_\text{PDE}} + \lambda\underbrace{\left[(x_\theta(0)-1)^2 + (x_\theta'(0))^2\right]}_{\mathcal{L}_\text{IC}}$$

The derivatives $x_\theta'$ and $x_\theta''$ are computed exactly via **automatic differentiation** (PyTorch autograd) — no finite differences, no grid.

```python
import torch

t = torch.linspace(0, 4*3.14159, 200).unsqueeze(1).requires_grad_(True)
x = net(t)
xd  = torch.autograd.grad(x.sum(),  t, create_graph=True)[0]   # x'(t)
xdd = torch.autograd.grad(xd.sum(), t, create_graph=True)[0]   # x''(t)

residual = xdd + omega**2 * x                                   # PDE residual
loss_pde = residual.pow(2).mean()
```

After ~5000 epochs of Adam, $x_\theta(t) \approx \cos(\omega t)$.

---

## 4. Understanding the loss function

```
L_total = L_pde  +  λ_bc · L_bc  +  λ_ic · L_ic  +  λ_data · L_data
```

| Term | Meaning | Typical weight |
|------|---------|---------------|
| `L_pde` | PDE residual at collocation points | 1.0 (reference) |
| `L_bc` | Boundary condition mismatch | 1–10 |
| `L_ic` | Initial condition mismatch | 10–200 |
| `L_data` | Mismatch to observations (optional) | 1–100 |

**Reading loss curves:**
- All terms should decrease together
- If `L_ic` stays high → increase `λ_ic`
- If `L_pde` oscillates → lower the learning rate
- If everything plateaus above 1e-3 → add more collocation points

**Relative L2 error:**

$$\varepsilon = \frac{\|u_\theta - u_\text{ref}\|_2}{\|u_\text{ref}\|_2}$$

- Getting started examples: $\varepsilon \sim 10^{-3}$ to $10^{-2}$
- Production-grade: $\varepsilon \sim 10^{-5}$

---

## 5. Running your first example

```bash
cd examples/getting_started
python 01_harmonic_oscillator.py
```

This trains a PINN to solve $x'' + \omega^2 x = 0$ for three different $\omega$ values and saves a comparison plot.

**What to look for in the output plot:**
- Solid line = exact solution $\cos(\omega t)$
- Dashed line = PINN prediction
- L2 error in the title

**Work through the examples in order:**

| File | Physics | What it teaches |
|------|---------|-----------------|
| `01_harmonic_oscillator.py` | $x'' + \omega^2 x = 0$ | Basic PINN setup |
| `02_damped_oscillator.py` | Damped ODE | Multiple regimes, exact solutions |
| `03_heat_diffusion_1d.py` | $u_t = \alpha u_{xx}$ | Hard BC enforcement |
| `04_wave_equation_1d.py` | $u_{tt} = c^2 u_{xx}$ | 2D spacetime domain |
| `05_logistic_growth.py` | $dN/dt = rN(1-N/K)$ | Nonlinear ODE, positivity |
| `06_lotka_volterra.py` | Predator-prey system | Multi-output PINN |
| `07_nonlinear_pendulum.py` | $\theta'' + \frac{g}{L}\sin\theta = 0$ | Nonlinearity, linearisation |
| `08_van_der_pol.py` | Van der Pol oscillator | Stiffness, limit cycles |
| `09_lorenz_system.py` | Lorenz chaos | PINN limits, chaos |
| `10_coupled_oscillators.py` | Two coupled masses | Normal modes, beating |

---

## 6. What to experiment with

After running an example as-is, try:

1. **Change `N_EPOCHS`** (e.g., 2000 vs 20000) — how does training length affect accuracy?
2. **Change a physics parameter** (e.g., `omega`, `alpha`, `c`) — does the network adapt?
3. **Change `N_COL`** (collocation points) — what is the minimum to get a good result?
4. **Remove `L_ic`** (set `λ_ic = 0`) — what happens? Why?
5. **Use a shallower network** (2 layers instead of 4) — capacity vs accuracy.

These micro-experiments build the intuition you need for Tier 2.

---

## 7. Where PINNs struggle

| Scenario | Why | What to use instead |
|----------|-----|---------------------|
| High-frequency solutions | Spectral bias of ReLU/Tanh nets | SIREN (sin activations), Fourier features |
| Long-time chaos | Error accumulates | Time-marching PINN (`templates/06_time_marching.py`) |
| Large 3D domains | Training cost scales poorly | FNO, DeepONet (Tier 2) |
| Sharp discontinuities / shocks | Smooth approximation fails | Shock-capturing PINN or classical solver |
| Pure data, no physics | Better tools exist | Standard ML, PyTorch Lightning |

Understanding these limits is what separates an Explorer from an Experimenter.

---

## 8. Milestones

Complete all of these before moving to Tier 2:

- [ ] Run all 10 examples in `examples/getting_started/` and understand each plot
- [ ] Modify at least 3 examples to explore a different physical regime
- [ ] Explain in your own words what the PDE residual loss means physically
- [ ] Understand why `λ_ic` is set to 100–200 in most examples
- [ ] Achieve $\varepsilon < 0.01$ on the heat equation example

**When you are ready:**

```python
import pinneaple_learning as pl
pl.learning_path(tier=2)
```
