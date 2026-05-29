# Getting Started Examples — Tier 1: Explorer

These examples are your **first step into Physics AI**. Each one:

- Solves a **classical physics problem** with a PINN
- Requires **only PyTorch + NumPy + Matplotlib** (no PINNeAPPle module imports needed)
- Runs in **under 5 minutes on CPU**
- Generates a **plot** comparing the PINN prediction against the exact or reference solution
- Includes **multiple parameter configurations** so you can see how physics changes the answer

Work through them in order. By example 10 you will have built intuition for:
- How PINNs encode physics as a loss function
- How initial and boundary conditions are enforced
- What kinds of problems PINNs solve well — and where they struggle

---

## Examples

| File | Problem | Key physics concept |
|------|---------|-------------------|
| [01_harmonic_oscillator.py](01_harmonic_oscillator.py) | Mass-spring: `x'' + ω²x = 0` | Natural frequency, period |
| [02_damped_oscillator.py](02_damped_oscillator.py) | Mass-spring-damper: `x'' + 2ζω x' + ω²x = 0` | Damping regimes: under / critical / over |
| [03_heat_diffusion_1d.py](03_heat_diffusion_1d.py) | Heat equation: `u_t = α u_xx` | Diffusivity, thermal decay |
| [04_wave_equation_1d.py](04_wave_equation_1d.py) | Wave equation: `u_tt = c² u_xx` | Wave speed, no decay |
| [05_logistic_growth.py](05_logistic_growth.py) | Logistic ODE: `dN/dt = rN(1-N/K)` | Carrying capacity, S-curve |
| [06_lotka_volterra.py](06_lotka_volterra.py) | Predator-prey: coupled nonlinear ODEs | Limit cycles, phase portrait |
| [07_nonlinear_pendulum.py](07_nonlinear_pendulum.py) | Pendulum: `θ'' + (g/L)sin(θ) = 0` | Nonlinear period lengthening |
| [08_van_der_pol.py](08_van_der_pol.py) | Van der Pol: `x'' − μ(1−x²)x' + x = 0` | Self-excited oscillation, stiffness |
| [09_lorenz_system.py](09_lorenz_system.py) | Lorenz: 3D chaotic system | Strange attractor, PINN limits |
| [10_coupled_oscillators.py](10_coupled_oscillators.py) | Two coupled masses + springs | Normal modes, beating phenomenon |

---

## How to run

```bash
cd examples/getting_started
python 01_harmonic_oscillator.py
```

Each script saves a PNG in the current directory.

---

## What to notice

**In every plot:**
- Black solid line = exact or reference solution
- Red dashed line = PINN prediction
- The L2 error in the title tells you how well the PINN matched

**Things to experiment with:**
- Change `N_EPOCHS` to see how training length affects accuracy
- Change the parameter values in `EXPERIMENTS` to explore new regimes
- Change `N_COL` (collocation points) to see how data density matters

---

## Next steps

After completing these examples, you are ready for **Tier 2 (Experimenter)**:
- [`templates/19_fno_neural_operator.py`](../../templates/19_fno_neural_operator.py) — learn solution operators
- [`templates/20_deeponet_surrogate.py`](../../templates/20_deeponet_surrogate.py) — parametric surrogates
- [`templates/28_arena_benchmark.py`](../../templates/28_arena_benchmark.py) — compare architectures systematically

Or go directly to **NVIDIA PhysicsNeMo** for production-scale problems:
- https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html
