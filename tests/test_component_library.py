"""Tests for pinneapple_systems.component_library."""
from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneapple_systems.component_library import (
    ComponentRegistry,
    ComponentModel,
    Physics,
    Toolbox,
    ToolboxRegistry,
    benchmark_component_type,
)


def _make_loader(in_dim: int, out_dim: int, n: int = 64, batch_size: int = 16) -> DataLoader:
    x = torch.rand(n, in_dim)
    y = torch.rand(n, out_dim)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                       collate_fn=lambda batch: {"x": torch.stack([b[0] for b in batch]),
                                                  "y": torch.stack([b[1] for b in batch])})


def test_registry_lists_reference_components():
    names = ComponentRegistry.list()
    for expected in ("pipevanillapinn", "pumpmodifiedmlp", "valvemodifiedmlp"):
        assert expected in names
    assert "pipe" in ComponentRegistry.component_types()
    assert len(ComponentRegistry.by_type("pipe")) >= 1


def test_build_assigns_physics_and_bookkeeping():
    model = ComponentRegistry.build("PipeVanillaPINN")
    assert isinstance(model, ComponentModel)
    assert model.physics is not None
    assert model.physics.name == "incompressible_flow"
    assert model._component_name == "pipevanillapinn"


def test_valve_has_no_physics_by_default():
    model = ComponentRegistry.build("ValveModifiedMLP")
    assert model.physics is None


def test_fit_and_evaluate_pipe():
    model = ComponentRegistry.build("PipeVanillaPINN", in_dim=3, out_dim=3, hidden=(16, 16))
    loader = _make_loader(3, 3, n=32, batch_size=8)
    result = model.fit(loader, loader, epochs=2, lr=1e-3, run_name="test_pipe", log_dir=tempfile.mkdtemp())
    assert "best_val" in result
    metrics = model.evaluate(torch.rand(10, 3), torch.rand(10, 3))
    assert set(metrics) == {"rmse", "mape", "r2"}


def test_fit_without_physics_valve():
    model = ComponentRegistry.build("ValveModifiedMLP", in_dim=2, out_dim=1, hidden_dim=16, n_layers=2)
    loader = _make_loader(2, 1, n=32, batch_size=8)
    result = model.fit(loader, loader, epochs=2, lr=1e-3, run_name="test_valve", log_dir=tempfile.mkdtemp())
    assert "best_val" in result


def test_checkpoint_round_trip_reconstructs_from_component_name():
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(8, 8))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hx.pt")
        model.save_checkpoint(path)
        restored = ComponentModel.load_checkpoint(path)
    assert restored._component_name == "heatexchangervanillapinn"
    assert restored._architecture == "vanilla_pinn"
    x = torch.rand(5, 2)
    out_a = model(x)
    out_b = restored(x)
    y_a = out_a.y if hasattr(out_a, "y") else out_a
    y_b = out_b.y if hasattr(out_b, "y") else out_b
    assert torch.allclose(y_a, y_b, atol=1e-6)


def test_physics_from_shortcut_unknown_raises():
    try:
        Physics.from_shortcut("not_a_real_shortcut")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_toolbox_registry_defaults_and_install():
    assert "fluid mechanics toolbox@1.0" in ToolboxRegistry.list()
    tb = ToolboxRegistry.install("Fluid Mechanics Toolbox")
    assert isinstance(tb, Toolbox)
    assert ToolboxRegistry.is_installed("Fluid Mechanics Toolbox", "1.0")
    assert len(tb.components()) >= 3
    ToolboxRegistry.uninstall("Fluid Mechanics Toolbox")
    assert not ToolboxRegistry.is_installed("Fluid Mechanics Toolbox", "1.0")


def test_benchmark_component_type_ranks_pipe_architectures():
    loader = _make_loader(3, 3, n=24, batch_size=8)
    result = benchmark_component_type("pipe", loader, loader, epochs=1)
    assert result.component_type == "pipe"
    assert len(result.ranked) >= 1
    best = result.best()
    assert best is not None and "final_loss" in best
