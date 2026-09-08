# PINN / Physics

This layer turns a [ProblemDefinition](problem_definition.md) and a
[Model](model.md) into a differentiable loss: it takes the model's
predictions at collocation points, differentiates them with autograd, and
assembles PDE-residual, boundary, and initial-condition penalties into a
single training objective.

## Where it lives

`pinneapple_physics.pinn_solver` (re-exported at `pinneapple_physics`):

- `compile_problem(spec, weights=None)` — the main entry point. Given a
  `ProblemSpec`, returns a `loss_fn(model, y_hat, batch)` callable that the
  `Trainer` calls every step. It reads `spec.pde.kind` to know which residual
  to build (`laplace`, `poisson`, `burgers`,
  `navier_stokes_incompressible`, `heat`, `wave`, `elasticity`, `darcy`,
  `helmholtz`, `advection`, `reaction_diffusion`, ...).
- `LossWeights` — the four fixed weights `compile_problem` reads directly:
  `w_pde`, `w_bc`, `w_ic`, `w_data`.
- `AdaptiveWeights` — a standalone self-normalizing weighter for arbitrary
  named loss terms (independent of `LossWeights`): it tracks a lagged EMA of
  each term's raw loss and rescales every term's weight relative to the
  currently *hardest* term, clamped to `[1, max_ratio]`.
- Autograd operators (`pinneapple_physics.pinn_solver.compiler.autograd_ops`):
  `grad(y, x)`, `jacobian(Y, x)`, `divergence(...)`, `laplacian(...)`,
  `time_derivative(y, x, t_index)`, `norm_dot_grad(y, x, normals)`, `mse`.
  These are the primitives every built-in residual (and any custom one) is
  built from.
- `Subdomain`, `SubdomainPINN`, `DoMINO` — domain-decomposition PINN
  (split the domain into subdomains, each with its own sub-network, stitched
  by interface conditions), for problems too stiff or too large for a single
  network.
- `LatentConditionedModel`, `sample_latent`, `ensemble_forward`,
  `mean_covariance_loss` — a stochastic/latent-conditioned PINN variant for
  predicting a distribution over solutions rather than a point estimate.

## Compiling and using a loss

```python
from pinneapple_physics import compile_physics  # wraps pinn_solver.compile_problem

losses = compile_physics(spec)          # spec: a ProblemSpec
# losses(model, y_hat, batch) -> {"pde": ..., "bc_<name>": ..., ...}
```

## Symbolic residuals

For PDEs not covered by a built-in `kind`, `pinneapple_physics.symbolic_pde`
compiles a residual straight from a SymPy expression:

```python
import sympy as sp
from pinneapple_physics.symbolic_pde import SymbolicPDE, HardBC

x, y = sp.symbols("x y")
u = sp.Function("u")
expr = u(x, y).diff(x, 2) + u(x, y).diff(y, 2) + 2*sp.pi**2*sp.sin(sp.pi*x)*sp.sin(sp.pi*y)
pde = SymbolicPDE(expr, coord_syms=[x, y], field_syms=[u])
residual_fn = pde.to_residual_fn(model)   # (N,1) PDE residual tensor
```

`HardBC` bakes a boundary condition into the network's output via a distance-
function ansatz (`wrap_model`) instead of penalizing it in the loss;
`PeriodicBC`/`MultiPeriodicBC`/`NeumannBC` cover other constraint styles.

## What this layer does *not* do

It does not train anything — it only produces loss values/tensors from a
model's predictions. Turning that loss into weight updates is the job of
[Solver](solver.md).
