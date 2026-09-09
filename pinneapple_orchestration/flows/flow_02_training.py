"""flow_02_training — train a registered component and log it to an
ExperimentStore."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pinneapple_registry import ExperimentStore

from .._prefect_compat import flow
from ..common.provenance import flow_run_manifest
from ..tasks import train_component_task


@flow(name="flow_02_training")
def flow_training(
    component_name: str,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    *,
    experiment_store: Optional[ExperimentStore] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    manifest_dir: str = "data/manifests",
    **build_kwargs: Any,
) -> Dict[str, Any]:
    with flow_run_manifest(
        "flow_02_training", {"component_name": component_name, "epochs": epochs, "lr": lr}, manifest_dir=manifest_dir
    ) as manifest:
        result = train_component_task(
            component_name, train_loader, val_loader,
            experiment_store=experiment_store, epochs=epochs, lr=lr, **build_kwargs,
        )
        manifest.log_task("train_component", "ok", best_val=result["fit_result"]["best_val"])
        manifest.add_artifact("best_val", result["fit_result"]["best_val"])
        return {**result, "run_id": manifest.run_id}
