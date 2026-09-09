"""pinneapple_orchestration — an optional Prefect-based pipeline layer
wiring together capabilities that already exist elsewhere in PINNeAPPle:
``pinneapple_registry`` (artifact storage), ``pinneapple_systems
.component_library`` (trainable components), and ``pinneapple_systems
.digital_twin``. No orchestration layer (Prefect, Airflow, or otherwise)
previously existed anywhere in this repo — confirmed via repo-wide grep.

Kept deliberately lean — five flows, not a large numbered pipeline suite,
since this is a generic library, not a single product with a fixed
vertical pipeline:

1. ``flow_data_generation``  — sample scenarios via a caller-supplied
   solver thunk, persist into ``DatasetStore``.
2. ``flow_training``         — build + train a registered component,
   log the run into ``ExperimentStore``.
3. ``flow_benchmark``        — rank every architecture registered for a
   ``component_type``.
4. ``flow_registry_publish`` — the terminal, human-confirmed stage
   promotion (``development`` -> ``staging``/``production``).
5. ``flow_digital_twin_prep``— build a live ``DigitalTwin`` from a
   registry-stored, stage-gated model.

Real Prefect (retries, scheduling, a UI, deployments) is available once
you ``pip install "pinneapple[orchestration]"``; without it, every flow
and task here is still a plain, directly callable Python function (see
``_prefect_compat.py``) — nothing in this package requires Prefect to be
usable or testable.

    from pinneapple_orchestration.flows import flow_training
    from pinneapple_registry import ArtifactRegistry

    registry = ArtifactRegistry("./my_registry")
    result = flow_training("PipeVanillaPINN", train_loader, val_loader,
                            experiment_store=registry.experiments, epochs=50)
"""
from __future__ import annotations

from ._prefect_compat import PREFECT_AVAILABLE
from . import tasks
from . import flows
from .common import flow_run_manifest, FlowRunManifest

__all__ = ["PREFECT_AVAILABLE", "tasks", "flows", "flow_run_manifest", "FlowRunManifest"]
