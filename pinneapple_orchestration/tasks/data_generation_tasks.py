"""pinneapple_orchestration.tasks.data_generation_tasks — the task behind
flow_01_data_generation.

Generic on purpose: solvers registered in ``pinneapple_simulation
.numerical_solvers.SolverRegistry`` take wildly different constructor and
call kwargs (an FDM heat solver's inputs look nothing like an SPH
particle solver's), so this task does not try to auto-configure an
arbitrary registered solver. Instead the caller supplies a zero-argument
``sample_fn() -> (coords, fields)`` thunk — in real use this would close
over a call like ``SolverRegistry.build("fdm", ...)(...)`` — and this
task's job is purely to run it N times and persist each run into a
``DatasetStore``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from pinneapple_registry import DatasetStore

from .._prefect_compat import task

SampleFn = Callable[[], Tuple[Any, Dict[str, Any]]]


@task
def generate_scenarios_task(
    dataset_store: DatasetStore,
    problem_id: str,
    sample_fn: SampleFn,
    n_scenarios: int,
    *,
    scenario_prefix: str = "scenario",
) -> List[str]:
    """Runs ``sample_fn()`` ``n_scenarios`` times and saves each result as
    a new scenario. Returns the list of saved scenario ids."""
    saved = []
    for i in range(n_scenarios):
        coords, fields = sample_fn()
        scenario_id = f"{scenario_prefix}_{i:04d}"
        dataset_store.save(problem_id, scenario_id, coords, fields)
        saved.append(scenario_id)
    return saved
