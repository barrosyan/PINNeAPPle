# Solver

In this framework's pipeline (Problem → Domain → Model → Physics → **Solver**
→ Backend → Researcher), "Solver" means the *training policy*: which
optimizer, how many stages, in what order, and with what scheduling —
distinct from `pinneapple_simulation.numerical_solvers`, which are classical
FEM/FDM/CFD solvers used to generate reference data (see
[Package Layers](../architecture/package_layers.md) for that distinction).

## Where it lives

The training-policy layer is `pinneapple_neural.trainer` (re-exported at
`pinneapple_neural`, and via the legacy alias `pinneapple_train`):

- `Trainer` + `TrainConfig` — the baseline policy: an Adam optimizer
  (`torch.optim.Adam`, configured by `TrainConfig.lr`/`weight_decay`), with
  optional gradient clipping, AMP, early stopping, checkpointing, and a
  `physics_aware_validation` flag that runs validation under
  `torch.enable_grad()` so PDE-residual validation losses (which need
  second derivatives) don't break inside a `no_grad` block.
- `TwoPhaseTrainer` + `TwoPhaseConfig` — a two-stage policy: Phase 1 fits the
  model to reference/measured data (supervised MSE), Phase 2 switches to the
  physics (PDE residual + BC) loss; `combined=True` instead ramps a physics
  weight in during Phase 1 rather than switching abruptly.
- `TimeMarchingTrainer` — splits `[0, T]` into sequential windows and trains
  a fresh copy of the network per window, using the previous window's
  prediction at its end time as the next window's initial condition (Wight &
  Zhao 2020) — for stiff or long-horizon temporal PDEs.
- `CausalPINNTrainer` + `CausalWeightScheduler` — weights collocation points
  by a decaying function of the cumulative residual at earlier times, so the
  network is pushed to get early times right before late ones (Wang et al.
  2022, "Respecting causality...").
- `DDPPINNTrainer` — the same training loop wrapped for
  `torch.nn.parallel.DistributedDataParallel`.
- `MultiRestartTrainer` — runs multiple random restarts and keeps the best.
- Loss-weight balancers, usable independently of which `Trainer` you pick:
  `WeightScheduler`/`SelfAdaptiveWeights`, `GradNormBalancer`,
  `LossRatioBalancer`, `NTKWeightBalancer`, `ReLoBRaLo`, `SoftAdapt`,
  `AugmentedLagrangian`, `InverseDirichlet`, `PCGrad`, `JointAdaptiveWeights`,
  `AutoBalancer`.

## Using it

```python
from pinneapple_neural import train_model  # Trainer shortcut

result = train_model(model, losses, epochs=5000)
```

`train_model` is a thin wrapper over `Trainer(model, loss_fn).fit(...)`; for
anything beyond the default single-stage Adam policy (two-phase, causal,
time-marching, distributed), construct the specific trainer class directly.

## What Solver does *not* do

It has no opinion on which physics or architecture it's optimizing — it
receives a model and a loss function and drives weight updates. It also
doesn't decide *where* those updates run (CPU/GPU, PyTorch/JAX); that's
[Backend](backend.md).
