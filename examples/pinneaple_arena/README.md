# PINNeAPPle Arena — Examples

These examples are meant to **show the full potential of `pinneaple_arena`**:

- Run an end-to-end benchmark (bundle → backend → metrics → artifacts → leaderboard)
- Swap backends (native / DeepXDE / JAX / PhysicsNeMo) without changing the task
- Add a **custom backend** that only exposes a `predict_fn`
- Add a **custom task + custom schema + synthetic bundle** (no changes to the core library)
- Do a small **hyperparameter sweep** and rank results in a leaderboard

## Prereqs

From the repo root:

```bash
pip install -e .
```

Optional backends:

```bash
pip install deepxde            # for 02_backend_swap_optional.py
pip install jax jaxlib         # for run_jax configs / JAX backend
# PhysicsNeMo: follow NVIDIA instructions (optional)
```

## Run

All scripts are runnable from the repo root:

```bash
python examples/arena/01_quickstart_native.py
python examples/arena/02_backend_swap_optional.py
python examples/arena/03_custom_predict_backend.py
python examples/arena/04_custom_task_poisson_bundle.py
python examples/arena/05_sweep_and_leaderboard.py
```

Artifacts are saved under `artifacts/examples/*`.

## What to look at

- `artifacts/examples/*/runs/<run_id>/summary.json` (key benchmark metrics)
- `artifacts/examples/*/leaderboard.json` (append-only, easy to plot)

