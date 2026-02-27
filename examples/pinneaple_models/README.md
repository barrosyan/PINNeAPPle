# Pinneaple Models — Showcase Examples

These examples are **curated demos** focused on the full breadth of `pinneaple_models`: registry/catalog usage, neural operators, PINNs (including inverse parameters), ROMs, GNNs, and time-series transformers.

## How to run

From the repo root:

```bash
python -m pip install -e .
# or, without editable install:
# set PYTHONPATH to repo root

python examples/pinneaple_models_showcase/00_registry_tour.py
python examples/pinneaple_models_showcase/10_operator_learning_fno_toy.py
python examples/pinneaple_models_showcase/20_pinn_inverse_parameter_ode.py
python examples/pinneaple_models_showcase/30_rom_pod_dmd.py
python examples/pinneaple_models_showcase/40_graph_gnn_message_passing.py
python examples/pinneaple_models_showcase/50_timeseries_transformer_forecast_toy.py
```

All scripts are designed to run in **seconds to a couple minutes** on CPU.

## What each script demonstrates

- `00_registry_tour.py` — unified `ModelRegistry`, `ModelCatalog`, and how to build models by name.
- `10_operator_learning_fno_toy.py` — a tiny neural operator training loop (FNO-1D) for a toy operator.
- `20_pinn_inverse_parameter_ode.py` — inverse parameter identification using `VanillaPINN.inverse_params`.
- `30_rom_pod_dmd.py` — POD compression + DMD rollout for a synthetic dynamical system.
- `40_graph_gnn_message_passing.py` — message passing GNN forward + masked loss.
- `50_timeseries_transformer_forecast_toy.py` — a transformer-based forecaster forward/training step on toy data.
