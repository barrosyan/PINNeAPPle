"""Tests for pinneapple_neural.physicsnemo_bridge.

``physicsnemo`` (a.k.a. ``nvidia-physicsnemo``, formerly ``nvidia-modulus``)
is not installed in this environment (checked at collection time below), so
the real "physicsnemo model in hand" path cannot be exercised here. These
tests therefore always cover:

  * the core optional-dependency contract -- importing
    ``pinneapple_neural.physicsnemo_bridge`` never requires physicsnemo;
  * the missing-dependency error path for ``require_physicsnemo_model``;
  * every non-physicsnemo-specific code path (the output-normalizing
    adapter, the UQ/digital-twin wiring, and the full active-learning
    retrain loop) exercised for real against plain ``torch.nn.Module``
    stand-ins -- these are genuine PhysicsNeMo-shaped models from
    PINNeAPPle's point of view (a trained physicsnemo model is, by the time
    it reaches this bridge, just an ``nn.Module``), so this is real
    coverage of the bridge logic, not a mock of it.

and additionally exercises a real ``physicsnemo``-backed model only when the
package happens to be present.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinneapple_data.active_learning import ActiveLearningConfig


def _physicsnemo_available() -> bool:
    try:
        import physicsnemo  # noqa: F401
        return True
    except ImportError:
        try:
            import modulus  # noqa: F401
            return True
        except ImportError:
            return False


PHYSICSNEMO_AVAILABLE = _physicsnemo_available()


# ---------------------------------------------------------------------------
# Import contract
# ---------------------------------------------------------------------------

def test_import_succeeds_without_physicsnemo():
    """Core contract: importing the bridge never requires physicsnemo."""
    import pinneapple_neural.physicsnemo_bridge as bridge

    expected = {
        "PhysicsNemoModelAdapter",
        "is_physicsnemo_available",
        "require_physicsnemo_model",
        "wrap_for_uq",
        "build_digital_twin_for_model",
        "RetrainConfig",
        "ALRoundResult",
        "ALRetrainResult",
        "active_learning_retrain_loop",
    }
    assert expected.issubset(set(bridge.__all__))


def test_is_physicsnemo_available_matches_real_environment():
    from pinneapple_neural.physicsnemo_bridge import is_physicsnemo_available

    assert is_physicsnemo_available() == PHYSICSNEMO_AVAILABLE


@pytest.mark.skipif(
    PHYSICSNEMO_AVAILABLE,
    reason="physicsnemo is installed; this test targets the missing-dependency error path",
)
def test_require_physicsnemo_model_raises_clear_error_without_physicsnemo():
    from pinneapple_neural.physicsnemo_bridge import require_physicsnemo_model

    with pytest.raises(ImportError, match="physicsnemo is required"):
        require_physicsnemo_model(nn.Linear(2, 2))


@pytest.mark.skipif(
    not PHYSICSNEMO_AVAILABLE,
    reason="physicsnemo not installed in this environment; cannot exercise the real import path",
)
def test_require_physicsnemo_model_accepts_nn_module_when_available():
    from pinneapple_neural.physicsnemo_bridge import require_physicsnemo_model

    model = nn.Linear(2, 2)
    assert require_physicsnemo_model(model) is model


def test_require_physicsnemo_model_rejects_non_module_when_available():
    if not PHYSICSNEMO_AVAILABLE:
        pytest.skip("requires physicsnemo installed to reach the type-check branch")
    from pinneapple_neural.physicsnemo_bridge import require_physicsnemo_model

    with pytest.raises(TypeError):
        require_physicsnemo_model(object())


# ---------------------------------------------------------------------------
# Stand-in models (physicsnemo-shaped from PINNeAPPle's point of view: a
# trained physicsnemo model is just an nn.Module by the time it reaches this
# bridge, so these exercise the exact same code paths).
# ---------------------------------------------------------------------------

class _PlainTensorModel(nn.Module):
    """Mirrors FNOSurrogate2D in examples/vs_physicsnemo/05.../example.py."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


class _DictOutputModel(nn.Module):
    """Mirrors a PhysicsNeMo-Sym-style model returning named fields."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, x):
        y = self.net(x)
        return {"u": y[:, 0], "v": y[:, 1], "p": y[:, 2]}


class _YAttrOutput:
    def __init__(self, y):
        self.y = y


class _YAttrModel(nn.Module):
    """Mirrors a PINNOutput/OperatorOutput-style wrapper object."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, x):
        return _YAttrOutput(self.net(x))


class _MissingFieldModel(nn.Module):
    def forward(self, x):
        return {"u": torch.zeros(x.shape[0])}


class _UnsupportedOutputModel(nn.Module):
    def forward(self, x):
        return "not a tensor"


# ---------------------------------------------------------------------------
# PhysicsNemoModelAdapter
# ---------------------------------------------------------------------------

def test_adapter_passes_through_plain_tensor_output():
    from pinneapple_neural.physicsnemo_bridge import PhysicsNemoModelAdapter

    model = _PlainTensorModel()
    adapter = PhysicsNemoModelAdapter(model, ["u", "v", "p"])
    x = torch.randn(5, 2)
    out = adapter(x)
    assert out.shape == (5, 3)
    assert torch.allclose(out, model(x))


def test_adapter_normalizes_dict_output_in_field_order():
    from pinneapple_neural.physicsnemo_bridge import PhysicsNemoModelAdapter

    model = _DictOutputModel()
    adapter = PhysicsNemoModelAdapter(model, ["u", "v", "p"])
    x = torch.randn(4, 2)
    out = adapter(x)
    assert out.shape == (4, 3)

    raw = model(x)
    expected = torch.stack([raw["u"], raw["v"], raw["p"]], dim=-1)
    assert torch.allclose(out, expected)


def test_adapter_normalizes_y_attribute_output():
    from pinneapple_neural.physicsnemo_bridge import PhysicsNemoModelAdapter

    model = _YAttrModel()
    adapter = PhysicsNemoModelAdapter(model, ["u", "v", "p"])
    x = torch.randn(3, 2)
    out = adapter(x)
    assert out.shape == (3, 3)


def test_adapter_raises_key_error_on_missing_field():
    from pinneapple_neural.physicsnemo_bridge import PhysicsNemoModelAdapter

    adapter = PhysicsNemoModelAdapter(_MissingFieldModel(), ["u", "v"])
    with pytest.raises(KeyError, match="missing field"):
        adapter(torch.randn(2, 2))


def test_adapter_raises_type_error_on_unsupported_output():
    from pinneapple_neural.physicsnemo_bridge import PhysicsNemoModelAdapter

    adapter = PhysicsNemoModelAdapter(_UnsupportedOutputModel(), ["u"])
    with pytest.raises(TypeError, match="Unsupported model output type"):
        adapter(torch.randn(2, 2))


# ---------------------------------------------------------------------------
# wrap_for_uq (FASE 2 of examples/vs_physicsnemo/05.../example.py)
# ---------------------------------------------------------------------------

def test_wrap_for_uq_plain_model_produces_uncertainty():
    from pinneapple_analysis.uncertainty import MCDropoutConfig
    from pinneapple_neural.physicsnemo_bridge import wrap_for_uq

    model = _PlainTensorModel()
    uq = wrap_for_uq(model, mc_dropout_config=MCDropoutConfig(n_samples=8, dropout_p=0.1))
    x = torch.randn(6, 2)
    result = uq.predict_with_uncertainty(x, n_samples=8, device=torch.device("cpu"))
    assert result.mean.shape == (6, 3)
    assert result.std.shape == (6, 3)
    assert torch.all(result.std >= 0)


def test_wrap_for_uq_dict_output_model_requires_field_names():
    from pinneapple_neural.physicsnemo_bridge import wrap_for_uq

    model = _DictOutputModel()
    uq = wrap_for_uq(model, field_names=["u", "v", "p"])
    x = torch.randn(5, 2)
    result = uq.predict_with_uncertainty(x, n_samples=6)
    assert result.mean.shape == (5, 3)


# ---------------------------------------------------------------------------
# build_digital_twin_for_model (FASE 3 of examples/vs_physicsnemo/05.../example.py)
# ---------------------------------------------------------------------------

def test_build_digital_twin_for_model_predicts_at_coords():
    from pinneapple_neural.physicsnemo_bridge import build_digital_twin_for_model

    model = _DictOutputModel()
    dt = build_digital_twin_for_model(
        model, field_names=["u", "v", "p"], coord_names=["x", "y"]
    )
    coords = {
        "x": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "y": np.array([0.0, 0.5, 1.0], dtype=np.float32),
    }
    pred = dt.predict(coords)
    assert set(pred.keys()) == {"u", "v", "p"}
    for arr in pred.values():
        assert arr.shape == (3,)


def test_build_digital_twin_for_model_with_plain_tensor_model():
    from pinneapple_neural.physicsnemo_bridge import build_digital_twin_for_model

    model = _PlainTensorModel()
    dt = build_digital_twin_for_model(model, field_names=["u", "v", "p"])
    coords = {"x": np.array([0.2, 0.8], dtype=np.float32), "y": np.array([0.1, 0.9], dtype=np.float32)}
    pred = dt.predict(coords)
    assert set(pred.keys()) == {"u", "v", "p"}


# ---------------------------------------------------------------------------
# active_learning_retrain_loop (Padrao 2)
# ---------------------------------------------------------------------------

class _SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def _target_fn(x_np: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x_np[:, 0]) * np.sin(np.pi * x_np[:, 1])


def _loss_fn(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    pred = model(xb).squeeze(-1)
    target = torch.sin(np.pi * xb[:, 0]) * torch.sin(np.pi * xb[:, 1])
    return torch.mean((pred - target) ** 2)


def _make_residual_fn(model: nn.Module):
    def _residual_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.from_numpy(x_np.astype(np.float32))
            pred = model(xt).squeeze(-1).numpy()
        return np.abs(pred - _target_fn(x_np))

    return _residual_fn


def test_active_learning_retrain_loop_structural_invariants():
    from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
        RetrainConfig,
        active_learning_retrain_loop,
    )

    torch.manual_seed(0)
    model = _SmallMLP()
    bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    al_cfg = ActiveLearningConfig(n_candidates=200, n_select=15, n_initial=40, seed=1)
    rc = RetrainConfig(n_rounds=3, n_epochs_per_round=5, batch_size=16, lr=1e-2, verbose=False)

    result = active_learning_retrain_loop(
        model, _loss_fn, _make_residual_fn(model), bounds,
        al_config=al_cfg, retrain_config=rc,
    )

    assert len(result.rounds) == 3
    # collocation set grows by exactly n_select points each round
    assert result.x_collocation.shape == (40 + 3 * 15, 2)
    for i, round_result in enumerate(result.rounds, start=1):
        assert round_result.round_index == i
        assert round_result.n_points_added == 15
        assert round_result.n_points_total == 40 + i * 15
        assert len(round_result.train_losses) == rc.n_epochs_per_round
    assert np.isfinite(result.final_loss)
    # points stay within the declared bounds
    assert np.all(result.x_collocation[:, 0] >= 0.0) and np.all(result.x_collocation[:, 0] <= 1.0)
    assert np.all(result.x_collocation[:, 1] >= 0.0) and np.all(result.x_collocation[:, 1] <= 1.0)


def test_active_learning_retrain_loop_actually_reduces_held_out_error():
    """End-to-end check that the loop is real training, not a no-op."""
    from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
        RetrainConfig,
        active_learning_retrain_loop,
    )

    torch.manual_seed(0)
    model = _SmallMLP()
    bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    residual_fn = _make_residual_fn(model)

    eval_pts = np.random.default_rng(42).random((500, 2)).astype(np.float32)
    err_before = float(residual_fn(eval_pts).mean())

    al_cfg = ActiveLearningConfig(n_candidates=500, n_select=30, n_initial=80, seed=1)
    rc = RetrainConfig(n_rounds=4, n_epochs_per_round=60, batch_size=32, lr=5e-3, verbose=False)

    active_learning_retrain_loop(
        model, _loss_fn, residual_fn, bounds, al_config=al_cfg, retrain_config=rc
    )

    err_after = float(residual_fn(eval_pts).mean())
    assert err_after < err_before * 0.5


def test_active_learning_retrain_loop_accepts_explicit_x_initial():
    from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
        RetrainConfig,
        active_learning_retrain_loop,
    )

    torch.manual_seed(0)
    model = _SmallMLP()
    bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    x_init = np.random.default_rng(0).random((25, 2)).astype(np.float32)
    al_cfg = ActiveLearningConfig(n_candidates=100, n_select=10, seed=2)
    rc = RetrainConfig(n_rounds=2, n_epochs_per_round=3, batch_size=8, verbose=False)

    result = active_learning_retrain_loop(
        model, _loss_fn, _make_residual_fn(model), bounds,
        x_initial=x_init, al_config=al_cfg, retrain_config=rc,
    )
    assert result.x_collocation.shape == (25 + 2 * 10, 2)
    np.testing.assert_allclose(result.x_collocation[:25], x_init)


def test_active_learning_retrain_loop_accepts_custom_optimizer():
    from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
        RetrainConfig,
        active_learning_retrain_loop,
    )

    torch.manual_seed(0)
    model = _SmallMLP()
    bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    al_cfg = ActiveLearningConfig(n_candidates=100, n_select=10, n_initial=30, seed=3)
    rc = RetrainConfig(n_rounds=2, n_epochs_per_round=3, batch_size=8, verbose=False)

    result = active_learning_retrain_loop(
        model, _loss_fn, _make_residual_fn(model), bounds,
        al_config=al_cfg, retrain_config=rc, optimizer=optimizer,
    )
    assert len(result.rounds) == 2


def test_retrain_config_and_result_dataclass_defaults():
    from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
        ALRetrainResult,
        ALRoundResult,
        RetrainConfig,
    )

    rc = RetrainConfig()
    assert rc.n_rounds == 5
    assert rc.selection_mode == "weighted"
    assert rc.device == "cpu"

    round_result = ALRoundResult(
        round_index=1, n_points_added=10, n_points_total=60, mean_train_loss=0.1
    )
    assert round_result.train_losses == []

    full = ALRetrainResult(rounds=[round_result], x_collocation=np.zeros((60, 2)), final_loss=0.05)
    assert full.rounds[0] is round_result
