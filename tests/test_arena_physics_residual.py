"""Tests for Arena's physics-residual capture and physics-aware ranking.

`pinneapple_arena.trainer.train_pinn` already computes a PDE-residual loss
term each step for PINN-family models (`_train_pinn_autograd`'s `r_loss`,
kept separate from the boundary-condition loss). These tests confirm that
value is now captured into `TrainResult.physics_residual` without changing
training behaviour, and that `pinneapple_arena.arena.physics_aware_rank`
correctly flags a model whose residual stayed high (because it was starved
of training epochs) while leaving the pure-accuracy ranking untouched.

Kept fast: a single small analytic PDE (Poisson 2D), tiny MLPs, and a
handful of collocation points — no full Arena.run() needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_arena.config import ModelConfig, NetworkConfig, TrainingConfig
from pinneapple_arena.model_factory import build_model
from pinneapple_arena.problems import get_problem
from pinneapple_arena.trainer import TrainResult, train_pinn
from pinneapple_arena.arena import rank_by_accuracy, physics_aware_rank


def _make_pinn_cfg(name: str, epochs: int, seed: int = 0) -> ModelConfig:
    return ModelConfig(
        name=name,
        type="vanilla_pinn",
        network=NetworkConfig(hidden=[16, 16]),
        training=TrainingConfig(epochs=epochs, lr=1e-2, grad_clip=1.0,
                                scheduler="none", seed=seed),
    )


@pytest.fixture(scope="module")
def poisson_setup():
    problem = get_problem("poisson_2d")
    rng = np.random.default_rng(0)
    xy_int, Y_int, xy_bc, Y_bc, xy_eval, Y_eval, field_names = problem.supervised_data(
        n_train=64, n_bc=32, grid_n=12)
    xy_col = rng.uniform(0, 1, (128, 2))
    # poisson_2d's physics_preset ("poisson_2d_default") doesn't resolve to a
    # real pinneapple_physics preset name, so this exercises the autograd
    # residual path (`_train_pinn_autograd`) — the one guaranteed to run in
    # practice for Arena's built-in problems today.
    assert problem.compiled_losses() is None
    return dict(problem=problem, xy_col=xy_col, xy_bc=xy_bc, Y_bc=Y_bc,
                xy_eval=xy_eval, Y_eval=Y_eval, field_names=field_names)


def _train(poisson_setup, name, epochs, seed=0):
    cfg = _make_pinn_cfg(name, epochs=epochs, seed=seed)
    model = build_model(cfg, in_dim=2, out_dim=1)
    return train_pinn(
        model, cfg,
        pinn_residuals_fn=poisson_setup["problem"].pinn_residuals,
        xy_int=poisson_setup["xy_col"],
        xy_bc=poisson_setup["xy_bc"],
        uv_bc=poisson_setup["Y_bc"],
        problem_params={},
    )


def test_train_result_has_physics_residual_field_defaulting_to_none():
    """New field exists and is opt-in — doesn't break old positional/keyword
    construction of TrainResult for non-PINN families."""
    import torch.nn as nn
    r = TrainResult(name="x", model=nn.Linear(1, 1), train_losses=[1.0], train_time=0.1)
    assert r.physics_residual is None


def test_pinn_training_populates_physics_residual(poisson_setup):
    result = _train(poisson_setup, "short_run", epochs=5)
    assert result.physics_residual is not None
    assert np.isfinite(result.physics_residual)
    assert result.physics_residual >= 0.0


def test_longer_training_yields_lower_or_equal_residual_windowed_average(poisson_setup):
    """Not a strict monotonicity guarantee (SGD is noisy) but a
    well-trained model should land at a much lower residual than a model
    given almost no epochs at all — this is the scenario the physics-aware
    flagging is meant to catch."""
    short = _train(poisson_setup, "short_run_2", epochs=3, seed=1)
    long = _train(poisson_setup, "long_run", epochs=400, seed=1)
    assert long.physics_residual < short.physics_residual


def test_physics_aware_rank_flags_high_residual_model(poisson_setup):
    short = _train(poisson_setup, "under_trained", epochs=3, seed=2)
    long_a = _train(poisson_setup, "well_trained_a", epochs=400, seed=2)
    long_b = _train(poisson_setup, "well_trained_b", epochs=400, seed=3)
    train_results = [short, long_a, long_b]

    # Fabricate eval metrics: give the under-trained model a deceptively
    # *good* accuracy score so the test actually exercises "physics-aware"
    # flagging rather than accuracy alone catching it.
    eval_results = [
        {"name": short.name,  "metrics": {"rel_u": 0.01}},
        {"name": long_a.name, "metrics": {"rel_u": 0.05}},
        {"name": long_b.name, "metrics": {"rel_u": 0.06}},
    ]
    field_names = ["u"]

    # Pure-accuracy ranking: unaffected by physics_residual, short_run wins.
    acc_ranking = rank_by_accuracy(train_results, eval_results, field_names)
    assert acc_ranking[0][0] == short.name

    view = physics_aware_rank(train_results, eval_results, field_names, n_std=0.5)
    # Ranking is identical to the pure-accuracy ranking (additive, not a
    # silent re-order).
    assert view["ranking"] == acc_ranking
    # But the deceptively "best" model is flagged for its high residual.
    assert short.name in view["flagged_high_residual"]
    assert long_a.name not in view["flagged_high_residual"]
    assert long_b.name not in view["flagged_high_residual"]
    assert set(view["physics_residuals"]) == {short.name, long_a.name, long_b.name}


def test_physics_aware_rank_with_no_residuals_flags_nothing():
    """Supervised/graph-only comparisons (physics_residual left None
    everywhere) must not crash or spuriously flag anything."""
    import torch.nn as nn
    r1 = TrainResult(name="a", model=nn.Linear(1, 1), train_losses=[1.0], train_time=0.1)
    r2 = TrainResult(name="b", model=nn.Linear(1, 1), train_losses=[1.0], train_time=0.1)
    eval_results = [
        {"name": "a", "metrics": {"rel_u": 0.1}},
        {"name": "b", "metrics": {"rel_u": 0.2}},
    ]
    view = physics_aware_rank([r1, r2], eval_results, ["u"])
    assert view["flagged_high_residual"] == []
    assert view["physics_residuals"] == {}
    assert view["group_median"] is None
