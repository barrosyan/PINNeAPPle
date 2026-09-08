# Researcher & Benchmarking

This is the outermost layer of the pipeline: once a model has been trained,
it orchestrates comparing it against other models/backends, tracking
experiments reproducibly, and (optionally) mining literature for reference
results to compare against.

## Benchmarking — `pinneapple_tools.benchmark_suite`

Re-exported at `pinneapple_tools` (legacy alias `pinneapple_arena`):

- `Arena` — the high-level runner connecting a `ProblemSpec` to the training
  pipeline. Build one via `Arena.from_spec(spec)` (or other class-method
  constructors), then call `.run(...)` or `.compare(...)`.
- `PINNArenaBenchmark` + `BenchmarkConfig`/`BenchmarkResult`/
  `BenchmarkTaskBase`/`ModelSpec` — the underlying multi-model benchmark
  engine; `DEFAULT_MODELS` gives a ready-made model set to compare.
- `TransferBenchmarkPipeline`/`MetaBenchmarkPipeline` (with matching
  `*Config`/`*Result` types) — benchmark harnesses specifically for transfer-
  learning and meta-learning scenarios.
- `TASK_REGISTRY`/`BACKEND_REGISTRY` (with `register_task`/`register_backend`/
  `get_task`/`get_backend`/`list_tasks`/`list_backends`) — pluggable
  registries so new benchmark tasks or execution backends can be added
  without modifying the runner.
- `BenchmarkReport`/`ModelRunResult` — structured run output, and
  `run_arena_experiment`/`run_full_pipeline` for driving a benchmark from a
  YAML/JSON experiment config end-to-end (see `pinneapple_arena` at the
  repo's top level for the YAML/JSON-driven runner and leaderboard).

```python
from pinneapple_tools import run_benchmark

results = run_benchmark({"my_pinn": model}, tasks=["burgers_1d"])
```

## Reproducibility — `pinneapple_neural.trainer.audit`

`RunLogger` writes a JSONL audit log per run (one line per logged record,
each timestamped); `set_seed`/`set_deterministic` fix RNG state and PyTorch
determinism flags so a run can be repeated. `TrainingAdvisor`
(`pinneapple_neural.trainer.advisor`) inspects a finished run's
`DiagnosticReport` and surfaces `Suggestion`s (e.g. loss-weight or LR
issues) for the next attempt.

## Literature pipeline — `pinneapple_tools.hpo_experiments`

Re-exported at `pinneapple_tools` (legacy alias `pinneapple_researcher`):
`discover(...)` finds papers for a given problem area, `build_kb(...)`
assembles a knowledge base from them, `extract_problem_solutions(...)` pulls
structured problem/solution pairs out of that knowledge base, and
`reproduce(...)` attempts to reproduce a published benchmark result —
useful for sanity-checking a new model or preset against literature before
trusting it.

## Named benchmarks — `pinneapple_pdb`

`pinneapple_pdb.benchmarks` provides a catalog of named reference
benchmarks: `BenchmarkEntry`, `register_benchmark`/`get_benchmark`/
`list_benchmarks`, and `benchmark_catalog`. This is what
`PhysicsGuardrail.check(reference_benchmark=...)` (in `pinneapple_llm`) and
similar consumers look up when they need to validate a result against a
named, versioned benchmark rather than an ad hoc reference solution.

## How the pieces fit

A typical loop is: run [Solver](solver.md) → get a trained model → hand it
to `Arena`/`PINNArenaBenchmark` alongside other models on the same task →
compare via `BenchmarkReport` → log the run with `RunLogger` for
reproducibility → optionally check literature agreement with
`hpo_experiments.reproduce` or a `pinneapple_pdb` named benchmark.
