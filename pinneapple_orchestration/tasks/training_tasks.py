"""pinneapple_orchestration.tasks.training_tasks — the task behind
flow_02_training. Thin: builds a component via ``ComponentRegistry`` and
delegates to ``ComponentModel.fit()`` (which itself delegates to the
shared ``Trainer``) — no training loop lives here. Also logs the run into
an ``ExperimentStore`` so it's queryable later independent of this flow's
own manifest file.
"""
from __future__ import annotations

import tempfile
from typing import Any, Dict, Optional

from pinneapple_registry import ExperimentStore
from pinneapple_systems.component_library import ComponentRegistry, ComponentModel

from .._prefect_compat import task


@task
def train_component_task(
    component_name: str,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    *,
    experiment_store: Optional[ExperimentStore] = None,
    epochs: int = 100,
    lr: float = 1e-3,
    log_dir: Optional[str] = None,
    **build_kwargs: Any,
) -> Dict[str, Any]:
    """Builds ``component_name`` via ``ComponentRegistry``, trains it, and
    (if ``experiment_store`` is given) logs the run. Returns a dict with
    the trained ``model`` plus ``Trainer.fit()``'s own result dict."""
    model: ComponentModel = ComponentRegistry.build(component_name, **build_kwargs)

    run_id = None
    if experiment_store is not None:
        run_id = experiment_store.start_run(component_name, name=f"flow_02_{component_name}")

    result = model.fit(
        train_loader, val_loader, epochs=epochs, lr=lr,
        run_name=f"orchestration_{component_name}",
        log_dir=log_dir or tempfile.mkdtemp(prefix="pinneapple_orchestration_"),
    )

    if experiment_store is not None and run_id is not None:
        experiment_store.log_metrics(run_id, {"best_val": result["best_val"]})
        experiment_store.finish_run(run_id, final_metrics={"best_val": result["best_val"]})

    return {"model": model, "fit_result": result, "run_id": run_id}
