# Execution Model

A run moves through the pipeline in this order. Each step is one or two
concrete calls, all of which are covered in depth in
[Core Concepts](../core_concepts/overview.md); this page is the map between
them.

1. **ProblemDefinition defines equations and constraints.**
   A `ProblemSpec` — built directly, via `ProblemBuilder`, or via
   `get_preset(name, **kwargs)` — pins down the PDE `kind`, coordinates,
   fields, boundary/initial conditions, and domain bounds.
   (`pinneapple_physics.pde_environment`)

2. **Domain/Geometry produces sampling.**
   `build_domain(name)` instantiates a `PhysicsDomain2D`/`3D`, and
   `sample_domain(domain, n_interior, n_boundary)` draws interior and
   boundary collocation point batches from it.
   (`pinneapple_design.geometry`)

3. **Model predicts fields from coords.**
   `build_model(name, **kwargs)` instantiates a `BaseModel` subclass (SIREN,
   ModifiedMLP, AFNO, MeshGraphNet, ...) via the model registry; its only
   job is `forward(x) -> fields`.
   (`pinneapple_neural.architectures`)

4. **Physics/PINN computes residuals and builds loss terms.**
   `compile_physics(spec)` (or `compile_problem` directly) turns the spec
   into a `loss_fn(model, y_hat, batch)` that differentiates the model's
   output with autograd (`grad`/`jacobian`/`divergence`/`laplacian`) to
   produce PDE-residual, boundary, and initial-condition loss terms,
   weighted by `LossWeights`.
   (`pinneapple_physics.pinn_solver`, `pinneapple_physics.symbolic_pde`)

5. **Solver applies an optimization strategy.**
   A `Trainer` (or `TwoPhaseTrainer`/`TimeMarchingTrainer`/
   `CausalPINNTrainer`/`DDPPINNTrainer` for more specialized policies) runs
   the optimization loop against the compiled loss, producing a trained
   model and a training history.
   (`pinneapple_neural.trainer`)

6. **Backend executes the run.**
   Tensor operations run through whichever backend is active
   (`set_backend("torch" | "jax")`); device placement, mixed precision, and
   distributed/HPC execution are handled by `pinneapple_neural.trainer`'s
   parallel/HPC utilities beneath that abstraction.
   (`pinneapple_tools.compute_backends`)

7. **Researcher evaluates metrics and stores artifacts.**
   The trained model is handed to `Arena`/`PINNArenaBenchmark` for
   comparison against other models or backends, `RunLogger` records a JSONL
   audit trail for reproducibility, and results can be checked against
   literature (`hpo_experiments.reproduce`) or a named benchmark
   (`pinneapple_pdb.benchmarks`).
   (`pinneapple_tools.benchmark_suite`, `pinneapple_tools.hpo_experiments`,
   `pinneapple_neural.trainer.audit`, `pinneapple_pdb`)

## What's optional

Steps 2 and 7 are the most commonly skipped or replaced in practice: a
preset's own `domain_bounds` is often enough without an explicit
`PhysicsDomain2D` (step 2), and a quick experiment may train and inspect a
model without ever touching the Arena/benchmark layer (step 7). Steps 1, 3,
4, and 5 are the minimum needed to train anything.
