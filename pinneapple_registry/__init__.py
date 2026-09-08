"""pinneapple_registry — a local, self-hosted artifact registry: versioned
model storage, dataset storage, experiment tracking, and problem-spec
history, plus Triton Inference Server export.

Why this exists alongside ``pinneapple_hub``: ``pinneapple_hub`` publishes
a *finished* model to the Hugging Face Hub (external storage, network +
auth required, one artifact type). Day-to-day work needs something that
runs entirely offline, versions every intermediate checkpoint (not just
the one you choose to publish), tracks datasets and experiment runs next
to those models, and can hand a model straight to a local Triton
Inference Server — none of which ``pinneapple_hub`` does or is meant to
do. The two are complementary: a model developed and versioned here can
still be pushed to the Hub via ``pinneapple_hub.push_to_hub`` once it's
ready to share.

    from pinneapple_registry import ArtifactRegistry

    registry = ArtifactRegistry("./my_registry")
    registry.models.save("pipe_flow", trained_pipe_model, metadata={"metrics": {"rmse": 0.02}})
    registry.models.promote("pipe_flow", registry.models.latest("pipe_flow"), "production")
    run_id = registry.experiments.start_run("pipe_flow")
    registry.experiments.log_metrics(run_id, {"val_loss": 0.01}, step=10)
"""
from __future__ import annotations

from .dataset_store import DatasetStore
from .experiment_store import ExperimentStore
from .model_store import ModelStore, VALID_STAGES
from .problem_store import ProblemStore
from .registry import ArtifactRegistry

__all__ = [
    "ArtifactRegistry",
    "ModelStore",
    "DatasetStore",
    "ExperimentStore",
    "ProblemStore",
    "VALID_STAGES",
]
