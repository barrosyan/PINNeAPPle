"""Tests for pinneapple_orchestration (all 5 flows, running as plain
functions since Prefect is not installed in this environment)."""
from __future__ import annotations

import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneapple_orchestration.flows import (
    flow_data_generation,
    flow_training,
    flow_benchmark,
    flow_registry_publish,
    flow_digital_twin_prep,
)
from pinneapple_orchestration import PREFECT_AVAILABLE
from pinneapple_registry import ArtifactRegistry
from pinneapple_systems.component_library import ComponentModel


def _registry() -> ArtifactRegistry:
    return ArtifactRegistry(tempfile.mkdtemp())


def _collate(batch):
    return {"x": torch.stack([b[0] for b in batch]), "y": torch.stack([b[1] for b in batch])}


def _loader(in_dim: int, out_dim: int, n: int = 32) -> DataLoader:
    x, y = torch.rand(n, in_dim), torch.rand(n, out_dim)
    return DataLoader(TensorDataset(x, y), batch_size=8, collate_fn=_collate)


def test_prefect_availability_flag_is_a_bool():
    assert isinstance(PREFECT_AVAILABLE, bool)


def test_flow_data_generation(tmp_path):
    reg = _registry()

    def sample_fn():
        return np.random.rand(20, 2).astype(np.float32), {"u": np.random.rand(20, 1).astype(np.float32)}

    result = flow_data_generation(reg.datasets, "pipe_flow", sample_fn, 3, manifest_dir=str(tmp_path / "manifests"))
    assert len(result["scenarios"]) == 3
    assert reg.datasets.list_scenarios("pipe_flow") == sorted(result["scenarios"])


def test_flow_training_logs_experiment(tmp_path):
    reg = _registry()
    loader = _loader(2, 1)
    result = flow_training(
        "HeatExchangerVanillaPINN", loader, loader,
        experiment_store=reg.experiments, epochs=2,
        in_dim=2, out_dim=1, hidden=(8, 8),
        manifest_dir=str(tmp_path / "manifests"),
    )
    assert isinstance(result["model"], ComponentModel)
    assert "best_val" in result["fit_result"]
    runs = reg.experiments.list_runs("HeatExchangerVanillaPINN")
    assert len(runs) == 1
    assert runs[0]["final_metrics"] is not None


def test_flow_benchmark(tmp_path):
    loader = _loader(3, 3)
    result = flow_benchmark("pipe", loader, loader, epochs=1, manifest_dir=str(tmp_path / "manifests"))
    assert result["result"].component_type == "pipe"
    assert len(result["result"].ranked) >= 1


def test_flow_registry_publish_requires_confirm(tmp_path):
    from pinneapple_systems.component_library import ComponentRegistry

    reg = _registry()
    model = ComponentRegistry.build("ValveModifiedMLP")

    try:
        flow_registry_publish(reg, "valve_demo", model, stage="production", confirm=False,
                               manifest_dir=str(tmp_path / "manifests"))
        assert False, "expected PermissionError"
    except PermissionError:
        pass

    result = flow_registry_publish(reg, "valve_demo", model, stage="production", confirm=True,
                                    manifest_dir=str(tmp_path / "manifests"))
    assert reg.models.metadata("valve_demo", result["version"])["stage"] == "production"


def test_flow_digital_twin_prep_gated_on_stage(tmp_path):
    from pinneapple_systems.component_library import ComponentRegistry

    reg = _registry()
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(8, 8))
    reg.models.save("hx_twin_demo", model, stage="development")

    try:
        flow_digital_twin_prep(reg, "hx_twin_demo", ["T"], ["x", "y"], manifest_dir=str(tmp_path / "manifests"))
        assert False, "expected PermissionError (still 'development', not 'production')"
    except PermissionError:
        pass

    v = reg.models.latest("hx_twin_demo")
    reg.models.promote("hx_twin_demo", v, "production")
    result = flow_digital_twin_prep(
        reg, "hx_twin_demo", ["T"], ["x", "y"], model_cls=ComponentModel,
        manifest_dir=str(tmp_path / "manifests"),
    )
    assert result["twin"] is not None
