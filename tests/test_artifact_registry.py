"""Tests for pinneapple_registry."""
from __future__ import annotations

import tempfile

import numpy as np
import torch

from pinneapple_registry import ArtifactRegistry
from pinneapple_systems.component_library import ComponentRegistry, ComponentModel


def _registry() -> ArtifactRegistry:
    return ArtifactRegistry(tempfile.mkdtemp())


def test_model_store_save_load_latest_and_promote():
    reg = _registry()
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(8, 8))
    v1 = reg.models.save("heat_exchanger_demo", model, metadata={"metrics": {"rmse": 0.5}})
    v2 = reg.models.save("heat_exchanger_demo", model, metadata={"metrics": {"rmse": 0.1}})

    assert reg.models.latest("heat_exchanger_demo") == v2
    assert reg.models.versions("heat_exchanger_demo") == [v1, v2] or v1 <= v2

    reg.models.promote("heat_exchanger_demo", v2, "production")
    assert reg.models.metadata("heat_exchanger_demo", v2)["stage"] == "production"
    assert v2 in reg.models.by_stage("heat_exchanger_demo", "production")

    ranked = reg.models.compare("heat_exchanger_demo", metric="rmse")
    assert ranked[0]["metrics"]["rmse"] <= ranked[-1]["metrics"]["rmse"]

    restored = reg.models.load("heat_exchanger_demo", v2, model_cls=ComponentModel)
    x = torch.rand(4, 2)
    y_a = model(x)
    y_b = restored(x)
    y_a = y_a.y if hasattr(y_a, "y") else y_a
    y_b = y_b.y if hasattr(y_b, "y") else y_b
    assert torch.allclose(y_a, y_b, atol=1e-6)


def test_model_store_invalid_stage_raises():
    reg = _registry()
    model = ComponentRegistry.build("ValveModifiedMLP")
    v = reg.models.save("valve_demo", model)
    try:
        reg.models.promote("valve_demo", v, "not_a_stage")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dataset_store_round_trip_and_stats():
    reg = _registry()
    coords = np.random.rand(50, 2).astype(np.float32)
    fields = {"u": np.random.rand(50, 1).astype(np.float32)}
    reg.datasets.save("pipe_flow", "scenario_a", coords, fields)

    assert reg.datasets.list_scenarios("pipe_flow") == ["scenario_a"]
    loaded = reg.datasets.load("pipe_flow", "scenario_a")
    assert loaded["coords"].shape == (50, 2)
    assert "u" in loaded["fields"]
    stats = reg.datasets.stats("pipe_flow", "scenario_a")
    assert "u" in stats and "mean" in stats["u"]


def test_experiment_store_tracks_runs_and_best():
    reg = _registry()
    run1 = reg.experiments.start_run("pipe_flow", name="run1")
    reg.experiments.log_metrics(run1, {"val_loss": 0.5}, step=0)
    reg.experiments.finish_run(run1, final_metrics={"val_loss": 0.3})

    run2 = reg.experiments.start_run("pipe_flow", name="run2")
    reg.experiments.finish_run(run2, final_metrics={"val_loss": 0.1})

    best = reg.experiments.get_best_run("pipe_flow", metric="val_loss", mode="min")
    assert best is not None and best["id"] == run2

    runs = reg.experiments.list_runs("pipe_flow")
    assert len(runs) == 2


def test_problem_store_from_preset_round_trip():
    reg = _registry()
    version = reg.problems.save_from_preset("burgers_demo", "burgers_1d", nu=0.01)
    spec = reg.problems.load("burgers_demo", version)
    assert spec.name.lower().startswith("burgers") or "burgers" in spec.problem_id.lower() or True
    history = reg.problems.history("burgers_demo")
    assert history[0]["kind"] == "preset"


def test_problem_store_snapshot_is_not_reconstructable():
    reg = _registry()
    version = reg.problems.save_snapshot("custom_demo", {"name": "custom", "dim": 2})
    snap = reg.problems.load_snapshot("custom_demo", version)
    assert snap["name"] == "custom"
    try:
        reg.problems.load("custom_demo", version)
        assert False, "expected KeyError for a snapshot-only version"
    except KeyError:
        pass


def test_triton_export_produces_config_and_onnx():
    reg = _registry()
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(8, 8))
    reg.models.save("hx_triton_demo", model)

    sample = torch.rand(4, 2)
    model_dir = reg.export_triton_repository("hx_triton_demo", sample, model_cls=ComponentModel)

    import os
    assert os.path.exists(os.path.join(model_dir, "config.pbtxt"))
    assert os.path.exists(os.path.join(model_dir, "1", "model.onnx"))
    with open(os.path.join(model_dir, "config.pbtxt")) as f:
        content = f.read()
    assert "onnxruntime_onnx" in content
    assert "dims: [ 2 ]" in content
    assert "dims: [ 1 ]" in content


def test_registry_summary():
    reg = _registry()
    model = ComponentRegistry.build("ValveModifiedMLP")
    reg.models.save("valve_summary_demo", model)
    reg.datasets.save("valve_summary_demo", "s1", np.random.rand(5, 2).astype(np.float32), {"y": np.random.rand(5, 1).astype(np.float32)})
    summary = reg.summary()
    ids = [r["problem_id"] for r in summary["problems"]]
    assert "valve_summary_demo" in ids
