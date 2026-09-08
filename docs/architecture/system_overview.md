# System Overview

**Problem → Data/Geometry → Model → Physics(PINN) → Solver → Backend → Research**

PINNeAPPle is organized as a chain of independent packages, one per stage of
this pipeline, connected by plain-data interfaces (`ProblemSpec`, sampled
coordinate tensors, a loss-function callable, a trained model) rather than
by inheritance or a shared runtime object. Each stage can be swapped without
touching the others: change the preset without touching the network, change
the network without touching the physics, change the training policy
without touching the physics, change PyTorch for JAX without touching the
training policy.

## The core packages

| Stage | Package |
|---|---|
| Problem definition | `pinneapple_physics.pde_environment` |
| Data container | `pinneapple_data` |
| Geometry & sampling | `pinneapple_design.geometry` |
| Model architectures | `pinneapple_neural.architectures` |
| Physics / residual compiler | `pinneapple_physics.pinn_solver`, `pinneapple_physics.symbolic_pde` |
| Training policy ("Solver") | `pinneapple_neural.trainer` |
| Execution backend | `pinneapple_tools.compute_backends` |
| Benchmarking / research | `pinneapple_tools.benchmark_suite`, `pinneapple_tools.hpo_experiments` |

See [Package Layers](package_layers.md) for the full mapping, including the
packages that sit *around* this core pipeline (classical solvers, UQ,
transfer/meta-learning, digital twins, design optimization, and more).

## A thin top-level convenience layer

The `pinneapple` top-level package re-exports the pieces most needed to get
started (`get_preset`, `build_model`, and similar) so a first experiment
doesn't require knowing which sub-package each piece lives in. It is a
convenience wrapper, not a separate implementation — every function it
exposes delegates to one of the packages above.

## Compatibility shims

Three packages exist purely to keep older import paths working while the
"real" implementation lives elsewhere: `pinneapple_models` re-exports
`pinneapple_neural.architectures`, `pinneapple_solvers` re-exports
`pinneapple_simulation.numerical_solvers`, and `pinneapple_train` re-exports
`pinneapple_neural.trainer`. New code should import from the non-shim
location directly.

## Why the separation matters

Because each stage only depends on the *interface* of the stage before it
(a `ProblemSpec`, a batch of coordinates, a model's `forward`, a loss
dict), you can, for example, benchmark the same trained model against a
classical numerical solver's reference solution
(`pinneapple_simulation.numerical_solvers`) without either package knowing
the other exists — they only share `PhysicalSample`/tensor data, not code.
This is also what makes the higher-layer packages (`pinneapple_analysis`
for UQ/validation/inverse problems, `pinneapple_adaptation` for transfer/
meta-learning, `pinneapple_systems` for time series/co-simulation/digital
twins, `pinneapple_design.design_optimizer` for shape optimization) additive
rather than invasive: they consume the core pipeline's outputs and add
capability on top, without the core pipeline needing to know they exist.
