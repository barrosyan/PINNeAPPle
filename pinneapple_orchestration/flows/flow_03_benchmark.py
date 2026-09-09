"""flow_03_benchmark — rank every architecture registered for a
component_type."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .._prefect_compat import flow
from ..common.provenance import flow_run_manifest
from ..tasks import benchmark_component_type_task


@flow(name="flow_03_benchmark")
def flow_benchmark(
    component_type: str,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    manifest_dir: str = "data/manifests",
) -> Dict[str, Any]:
    with flow_run_manifest(
        "flow_03_benchmark", {"component_type": component_type, "epochs": epochs}, manifest_dir=manifest_dir
    ) as manifest:
        result = benchmark_component_type_task(component_type, train_loader, val_loader, epochs=epochs, lr=lr)
        manifest.log_task("benchmark", "ok", ranked=[r["name"] for r in result.ranked])
        manifest.add_artifact("ranked", result.ranked)
        return {"result": result, "run_id": manifest.run_id}
