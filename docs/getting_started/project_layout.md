# Project Layout

PINNeAPPle's `pinneapple_*` packages map to framework layers — see
[Architecture → Package Layers](../architecture/package_layers.md). This
page is the map of everything *around* those packages: the top-level
directories that hold the web app, runnable code samples, operational
scripts, and example projects.

## `pinneapple_app/`

The FastAPI + frontend web application for benchmarking PINN models through
a browser instead of a script. `backend/` is a FastAPI service
(`main.py`, `routers/`, `core/`); `frontend/` is a Vite/React app
(`src/`, `index.html`, `tailwind.config.js`). `Dockerfile.backend`,
`Dockerfile.frontend`, `docker-compose.yml`, and `nginx.conf` compose the
two into a deployable service. This is the only directory in the repo meant
to run as a long-lived server rather than a one-off script.

## `examples/`

Runnable scripts organized **by package/topic** (`architectures/`,
`data_pipeline/`, `geometry/`, `numerical_solvers/`, `pde_environment/`,
`pinn_solver/`, `problem_designer/`, `trainer/`, `time_series/`,
`benchmark_suite/`, `hpo_experiments/`, `arena_pipelines/`, `systems/`,
`worldmodel/`, `use_cases/`, `visualizations/`, `end_to_end/`, plus
comparison folders like `vs_physicsnemo/` and
`pinneapple_and_physicsnemo/`), meant to be run from the repo root, e.g.
`python examples/data_pipeline/01_physical_sample_basics.py` (see
`examples/README.md`). Also holds a few standalone results-generation
scripts (`generate_chassis_results.py`, `embed_results.py`,
`export_pdf.py`) and their output folders (`_out/`, `_runs*/`).

## `templates/`

A curated, numbered set of ~35 standalone scripts (`01_basic_pinn.py`
through `35_jax_backend.py`), each demonstrating **one capability
end-to-end** in a single file: domain decomposition, causal training, CSG
geometry, shape optimization, CAD-to-CFD, meta-learning, world models,
rigid-body/MPM simulation, uncertainty quantification, transfer learning,
inverse problems, neural operators (FNO/DeepONet), active learning, model
serving/export, digital twins, GNN mesh models, ROMs, time-series models,
Arena benchmarking, Koopman autoencoders, Zarr data pipelines, LLM-assisted
problem design, physics validation, RANS turbulence, 3D heat conduction, and
the JAX backend. Where `examples/` is organized by package and includes
multi-step pipelines, `templates/` is organized by *feature* — one file, one
concept, meant to be read top-to-bottom as a reference.

## `scripts/`

Operational runbook scripts for a specific end-to-end MVP flow (a 2D flow-
past-obstacle case), documented in `scripts/README.md`: `cosmos/` (optional
scene-to-spec generation via a Cosmos endpoint), `omniverse/` (USD
generation and point sampling, run inside Omniverse Kit), `mvp/` (bundle
validation and local Arena matrix runs), `arena/` (benchmark run +
leaderboard), and `dev/` for miscellaneous development helpers, plus two
top-level utilities: `update_imports.py` and `validate_model_card.py`. This
is a runbook for one concrete pipeline, not a general-purpose scripts
folder.

## `tools/`

Repository build tooling. Currently just `gen_api_docs.py` — the
`mkdocs-gen-files` script that generates `docs/api/` and `docs/SUMMARY.md`'s
API navigation section from the `pinneapple_*` packages' docstrings at
`mkdocs build` time (see [API Reference](../api/index.md) and
[Package Layers](../architecture/package_layers.md)).

## `projects/`

A worked example of a full physics project built on top of the framework,
rather than a framework component itself: `main.py` runs an 8-step
pipeline (geometry generation → point sampling from an STL → scenario spec
generation → OpenFOAM sampling across scenarios → bundle export → PINN
training → operator-model training → evaluation/ranking) for the
`heat_channel_obstacle_3d/` case, driven by `configs/` and
`scripts/pipeline/`. Use this directory as a template for structuring a new
project, not as library code to import from.

## `notebooks/`

Jupyter notebooks for interactive walkthroughs, e.g.
`PINNeAPPle_Demo_Heat3D_Laplace.ipynb` — useful for exploring a result
interactively after a script-based run, less suited to automated pipelines.

## `benchmarks/`

Micro-benchmarks of the *library's own* performance and internals — not
physics benchmarks. `data_io_bench.py` and `shard_balance_bench.py` measure
`pinneapple_data` I/O and sharding performance; `_latency.py` measures call
overhead. `_out/` holds checked-in results (CSV/JSON/plots) from a prior
run. Contrast with `pinneapple_tools.benchmark_suite`
(see [Researcher & Benchmarking](../core_concepts/researcher_benchmarking.md)),
which benchmarks *models*, not the library itself.
