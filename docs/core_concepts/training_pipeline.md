# Training Pipeline

**Problem → Domain → Model → Physics → Solver → Backend → Metrics.**

Each arrow is a real, swappable API boundary — not just a diagram. This
page shows the same pipeline at two levels: the explicit building blocks,
and the one-call convenience wrapper built on top of them.

## Building it explicitly

```python
from pinneapple_physics import get_preset, compile_physics
from pinneapple_design import build_domain, sample_domain
from pinneapple_neural import build_model, train_model

# 1. Problem — what to solve
spec = get_preset("burgers_1d", nu=0.01)

# 2. Domain — where to sample (only needed if you're not using the
#    preset's own domain_bounds directly, e.g. for a custom geometry)
domain = build_domain("channel_2d")
x_int, x_bnd = sample_domain(domain, n_interior=4096, n_boundary=512)

# 3. Model — coordinates -> fields
model = build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)

# 4. Physics — residual + constraint losses, compiled from the spec
losses = compile_physics(spec)

# 5. Solver — training policy (Adam, single stage, here)
result = train_model(model, losses, epochs=5000)

# 6. Backend — which runtime executed the above (PyTorch by default;
#    pinneapple_tools.compute_backends.set_backend("jax") switches it)

# 7. Metrics / Researcher — compare, log, and validate the result
```

Each numbered step maps onto one of the [Core Concepts](overview.md) pages;
swapping any one component (a different preset, domain, architecture,
trainer, or backend) does not require touching the others.

## The top-level `pinneapple` convenience package

`import pinneapple as pp` re-exports the pieces most used at the start of an
experiment — `pp.get_preset`/`pp.list_presets`/`pp.register_preset` (step 1)
and `pp.build_model` (step 3) are real top-level wrappers over the same
`pinneapple_physics`/`pinneapple_neural` functions used above. It does
*not* currently wrap steps 4–5 (physics compilation / training) — for
those, import `compile_physics`/`train_model` from `pinneapple_physics`/
`pinneapple_neural` directly as shown above.

`pp.quickstart(problem_id, **problem_kwargs)` is a summary helper, not a
full pipeline run: it loads the named preset and prints its PDE kind,
fields, coordinates, domain bounds, conditions, and configured solver, then
returns the `ProblemSpec` — useful as a first sanity check on a preset
before writing the explicit pipeline above, but it does not sample a
domain, train a model, or produce a visualization by itself.

## Where each stage can diverge from this default path

- **Problem**: hand-build a `ProblemSpec`/`ProblemBuilder` instead of a
  preset — see [ProblemDefinition](problem_definition.md).
- **Domain**: use a CSG/SDF-composed or mesh-based domain instead of a
  built-in one — see [Geometry & Domain](geometry_domain.md).
- **Physics**: use `SymbolicPDE` for a PDE with no built-in `kind`, or
  `DoMINO`/`SubdomainPINN` for domain decomposition — see
  [PINN / Physics](pinn.md).
- **Solver**: use `TwoPhaseTrainer`, `TimeMarchingTrainer`,
  `CausalPINNTrainer`, or `DDPPINNTrainer` instead of the default `Trainer`
  — see [Solver](solver.md).
- **Backend**: switch to JAX for the residual computation — see
  [Backend](backend.md).
- **Metrics / Researcher**: run the trained model through `Arena`/
  `PINNArenaBenchmark` instead of inspecting it standalone — see
  [Researcher & Benchmarking](researcher_benchmarking.md).
