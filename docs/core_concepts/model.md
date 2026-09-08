# Model

A **Model** maps coordinates to predicted fields — `u = model(x)`. It knows
nothing about the PDE it is being used to solve; physics enters later, as a
loss computed *on top of* the model's output. This separation is what lets
you swap a SIREN for a Fourier neural operator without touching the physics
code, and swap the PDE without touching the model code.

## Where it lives

Architectures live in `pinneapple_neural.architectures` (re-exported at
`pinneapple_neural` and, for legacy imports, `pinneapple_models`). Every
model subclasses `BaseModel(nn.Module)`
(`pinneapple_neural/architectures/base.py`), which adds framework-wide
conventions on top of a normal `torch.nn.Module`:

- `forward(x)` — the actual coordinate → field mapping (subclass-defined).
- `forward_batch(batch)` — a default dict-based entry point used by the
  `Trainer`/Arena: it reads `batch["x"]` (or `batch["x_col"]` for PINN
  collocation batches) and calls `forward`.
- `save_checkpoint(path, metadata=None)` / a matching load path — writes a
  dict with `state_dict`, `class_name`, and user `metadata` so a checkpoint
  can be reloaded without knowing which class produced it ahead of time.
- `family` / `name` class attributes used by the model registry for
  discovery and reporting.

## Building a model

Rather than importing architecture classes directly, use the registry:

```python
from pinneapple_neural import build_model

model = build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
```

`build_model` is a thin wrapper over `ModelRegistry.build`
(`pinneapple_neural.architectures.registry`), which looks the name up in the
`ModelCatalog` and instantiates it with the given kwargs — the same registry
the benchmark suite (`pinneapple_tools.benchmark_suite`) uses to build every
model in a comparison run from a plain string name.

## Available architecture families

`pinneapple_neural.architectures` ships several families, all subclassing
`BaseModel`: PINN-style MLPs (`SIREN`, `ModifiedMLP` with Fourier feature
embeddings), coordinate encodings (`HashGridMLP`), neural operators (`AFNO`
and others under `neural_operators`), graph networks (`MeshGraphNet`), plus
transformer, ROM, reservoir-computing, and classical time-series model
families under the same registry. Which family you pick affects only how
coordinates map to fields — everything downstream (loss compilation,
training, benchmarking) is architecture-agnostic.

## What a Model is *not*

A `Model` does not know about boundary conditions, PDE residuals, or
optimizers. Those responsibilities belong to
[ProblemDefinition](problem_definition.md), [PINN / Physics](pinn.md), and
[Solver](solver.md) respectively. The only contract a `Model` has to satisfy
is: given a batch of coordinates, return a batch of predicted field values.
