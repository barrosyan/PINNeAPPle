"""flow_01_data_generation — sample N scenarios via a caller-supplied
solver thunk and persist them into a DatasetStore."""
from __future__ import annotations

from typing import Any, Dict, List

from pinneapple_registry import DatasetStore

from .._prefect_compat import flow
from ..common.provenance import flow_run_manifest
from ..tasks import generate_scenarios_task
from ..tasks.data_generation_tasks import SampleFn


@flow(name="flow_01_data_generation")
def flow_data_generation(
    dataset_store: DatasetStore,
    problem_id: str,
    sample_fn: SampleFn,
    n_scenarios: int,
    *,
    manifest_dir: str = "data/manifests",
) -> Dict[str, Any]:
    with flow_run_manifest(
        "flow_01_data_generation", {"problem_id": problem_id, "n_scenarios": n_scenarios}, manifest_dir=manifest_dir
    ) as manifest:
        scenarios: List[str] = generate_scenarios_task(dataset_store, problem_id, sample_fn, n_scenarios)
        manifest.log_task("generate_scenarios", "ok", n_scenarios=len(scenarios))
        manifest.add_artifact("scenarios", scenarios)
        return {"problem_id": problem_id, "scenarios": scenarios, "run_id": manifest.run_id}
