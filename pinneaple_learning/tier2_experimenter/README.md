# Tier 2 — Experimenter: Building Physics AI Experiments

> **Who this is for:** You have run at least one PINN before and want to build systematic experiments, compare architectures, solve inverse problems, and quantify uncertainty.

---

## Table of contents

1. [The Experimenter mindset](#1-the-experimenter-mindset)
2. [Architecture comparison](#2-architecture-comparison)
3. [Running an inverse problem](#3-running-an-inverse-problem)
4. [Uncertainty quantification](#4-uncertainty-quantification)
5. [Active learning](#5-active-learning)
6. [Transfer learning](#6-transfer-learning)
7. [Benchmarking with Arena](#7-benchmarking-with-arena)
8. [Physics validation](#8-physics-validation)
9. [Milestones before advancing to Tier 3](#9-milestones)

---

## 1. The Experimenter mindset

At Tier 1 you ran a PINN and watched it converge. At Tier 2 you ask:

- *Is PINN the right tool here, or would FNO / DeepONet be better?*
- *How confident is the model? Where does it fail?*
- *Can I recover the unknown diffusivity from sparse measurements?*
- *Is my training efficient, or am I wasting compute on easy regions?*

The Experimenter builds **controlled experiments** — same problem, multiple approaches, reproducible metrics.

---

## 2. Architecture comparison

Four main architecture families for Physics AI:

| Architecture | When to use | PINNeAPPle template |
|-------------|-------------|---------------------|
| **PINN** | Single instance, inverse problems, irregular geometry | `templates/01_basic_pinn.py` |
| **FNO** | Parametric families on regular grids, fast inference | `templates/19_fno_neural_operator.py` |
| **DeepONet** | Forcing/control → response problems | `templates/20_deeponet_surrogate.py` |
| **MeshGraphNet** | Unstructured meshes, FEM surrogates | `templates/25_gnn_mesh.py` |

**Quick comparison code:**

```python
from pinneaple_arena import ArenaRunner, NativeBenchmarkTask

runner = ArenaRunner(task=NativeBenchmarkTask("poisson_1d"))
results = runner.run_all()
results.leaderboard()   # accuracy × latency × memory table
```

See [`templates/28_arena_benchmark.py`](../../templates/28_arena_benchmark.py)

---

## 3. Running an inverse problem

An **inverse problem** recovers unknown parameters in the PDE from observations.

**Example:** recover diffusivity $\alpha$ from temperature measurements.

```python
import torch
from pinneaple_uq import EKISolver  # Ensemble Kalman Inversion

# Unknown parameter as a trainable tensor
alpha = torch.nn.Parameter(torch.tensor(0.5))

# Hybrid loss: physics + data
loss = loss_pde(net, alpha) + 100 * loss_data(net, x_obs, u_obs)
```

The gradient flows through both the network weights AND `alpha` simultaneously.

**Gradient-free alternative** — EKI:

```python
solver = EKISolver(forward_model=forward, n_ensemble=50)
alpha_est = solver.run(observations=u_obs)
```

See [`templates/18_inverse_problem.py`](../../templates/18_inverse_problem.py)

---

## 4. Uncertainty quantification

Three approaches, increasing in cost and reliability:

**MC Dropout** (cheapest):
```python
from pinneaple_uq import MCDropoutPredictor

# Add dropout layers to your network
net = build_net(dropout_rate=0.1)
predictor = MCDropoutPredictor(net, n_samples=100)
mean, std = predictor.predict(x_test)
```

**Deep Ensemble** (most reliable):
```python
from pinneaple_uq import DeepEnsemble

ensemble = DeepEnsemble(build_fn=build_net, n_models=5)
ensemble.train(x_train, y_train, loss_fn)
mean, std = ensemble.predict(x_test)
```

**Conformal Prediction** (coverage guarantee, post-hoc):
```python
from pinneaple_uq import ConformalPredictor

cp = ConformalPredictor(net, coverage=0.90)
cp.calibrate(x_cal, y_cal)
lower, upper = cp.predict_interval(x_test)
```

See [`templates/16_uncertainty_quantification.py`](../../templates/16_uncertainty_quantification.py)

---

## 5. Active learning

Instead of placing collocation points uniformly, focus on regions where the PDE residual or prediction uncertainty is highest.

```python
from pinneaple_train import ResidualActiveSampler, VarianceActiveSampler

# Residual-based: add points where |F[u_θ]| is large
sampler = ResidualActiveSampler(net, pde_residual_fn, budget=2000)
t_new = sampler.sample(t_current)

# Variance-based: add points where MC Dropout std is large
sampler = VarianceActiveSampler(net, budget=2000, n_mc=30)
t_new = sampler.sample(t_current)
```

Typical result: 40–60% fewer collocation points for the same accuracy.

See [`templates/21_active_learning.py`](../../templates/21_active_learning.py)

---

## 6. Transfer learning

If you solved a related problem (different $\alpha$, same PDE family), reuse the weights:

```python
from pinneaple_transfer import FineTuner, ParametricFreezer

# Freeze early layers, fine-tune only the last two
finetuner = FineTuner(
    pretrained_net=source_net,
    freeze_strategy=ParametricFreezer(freeze_until_layer=2),
)
finetuner.train(new_loss_fn, n_epochs=2000)
```

Speedup: 3–10× fewer epochs compared to training from scratch.

See [`templates/17_transfer_learning.py`](../../templates/17_transfer_learning.py)

---

## 7. Benchmarking with Arena

The Arena runs multiple models on the same problem and produces a reproducible leaderboard.

```bash
python templates/28_arena_benchmark.py
```

Output:
- `28_arena_benchmark_result.png` — accuracy vs inference time scatter
- `benchmark_results.csv` — full metrics table

Use this to justify your architecture choice with data, not intuition.

---

## 8. Physics validation

A model that looks good on L2 error can still violate conservation laws.

```python
from pinneaple_validate import PhysicsValidator, ConservationLawChecker

validator = PhysicsValidator(net, pde_residual_fn)
report = validator.full_report(x_test)

checker = ConservationLawChecker(net)
checker.check_energy_conservation(x_test, u_test)
```

Always run validation before claiming a model is ready.

See [`templates/32_physics_validation.py`](../../templates/32_physics_validation.py)

---

## 9. Milestones

Complete all of these before moving to Tier 3:

- [ ] Compare PINN, FNO, and DeepONet on the same 1D problem using Arena
- [ ] Solve an inverse problem and recover a physical parameter to within 5%
- [ ] Quantify uncertainty with at least one method and check calibration
- [ ] Run active learning and show fewer points achieve the same accuracy
- [ ] Validate that your best model satisfies the relevant conservation law
- [ ] Achieve $\varepsilon < 10^{-3}$ on a 2D PDE (heat or Poisson)

**When you are ready:**

```python
import pinneaple_learning as pl
pl.learning_path(tier=3)
```
