"""Tests for pinneapple_arena.nas — search-based architecture + hyperparameter
selection over the existing ModelRegistry catalog.

Not a differentiable/weight-sharing NAS (see nas.py's module docstring) —
each trial trains an independent small model from scratch via the real
`train_pinn` path and is scored via the real `evaluate_model` metrics
(+ optionally the real `TrainResult.physics_residual`). Kept fast: a small
analytic PDE (poisson_2d, same problem `test_arena_physics_residual.py`
uses), tiny epoch budgets, a restricted 2-architecture search space, and a
handful of trials — this is a test of the search *mechanism*, not a real
hyperparameter optimization run.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_arena.nas import (
    ArchitectureCandidate,
    ArchitectureSearchSpace,
    DEFAULT_CANDIDATES,
    NASResult,
    search_architecture,
)
from pinneapple_arena.problems import get_problem


# Restrict to two of the three default candidates, kept tiny, so the test
# runs quickly while still exercising real architecture-selection (not just
# hyperparameter selection within a single fixed architecture).
def _tiny_search_space() -> ArchitectureSearchSpace:
    return ArchitectureSearchSpace(
        candidates={
            "vanilla_pinn": ArchitectureCandidate(
                name="vanilla_pinn",
                param_space={
                    "n_layers": ("int", (2, 3)),
                    "width": ("int", (8, 16)),
                    "activation": ("categorical", (["tanh", "silu"],)),
                },
                build_kwargs_fn=DEFAULT_CANDIDATES["vanilla_pinn"].build_kwargs_fn,
            ),
            "siren": ArchitectureCandidate(
                name="siren",
                param_space={
                    "n_layers": ("int", (2, 3)),
                    "hidden_dim": ("int", (8, 16)),
                    "omega_0": ("float", (10.0, 30.0)),
                },
                build_kwargs_fn=DEFAULT_CANDIDATES["siren"].build_kwargs_fn,
            ),
        },
    )


@pytest.fixture(scope="module")
def poisson_problem():
    return get_problem("poisson_2d")


def test_search_architecture_returns_valid_result(poisson_problem):
    space = _tiny_search_space()
    result = search_architecture(
        poisson_problem, search_space=space, n_trials=5,
        epochs_per_trial=8, n_train=32, n_bc=16, n_col=48, grid_n=8,
        seed=0,
    )

    assert isinstance(result, NASResult)
    assert result.best_architecture in space.architecture_names
    assert np.isfinite(result.best_score)
    assert len(result.all_trials) == 5
    for t in result.all_trials:
        assert t["architecture"] in space.architecture_names
        assert isinstance(t["hyperparams"], dict)
        assert "lr" in t["hyperparams"]
        # every trial in this search space trains a PINN-family model
        # through train_pinn, so a physics_residual should always be
        # captured (successful trials only — an errored trial has no
        # "physics_residual" key).
        if t.get("score") is not None:
            assert t["physics_residual"] is not None
            assert np.isfinite(t["physics_residual"])

    # best_hyperparams should be exactly the hyperparams of the winning trial.
    best_trials = [t for t in result.all_trials if t.get("score") == result.best_score]
    assert best_trials
    assert result.best_hyperparams == best_trials[0]["hyperparams"]


def test_search_architecture_default_space_names_are_registered():
    """The default candidate names must actually resolve in ModelRegistry —
    guards against the search space silently drifting from the registry
    (e.g. a rename)."""
    from pinneapple_neural.architectures import ModelRegistry

    space = ArchitectureSearchSpace()
    assert set(space.architecture_names) == {"vanilla_pinn", "modified_mlp", "siren"}
    for name in space.architecture_names:
        ModelRegistry.spec(name)  # raises KeyError if not registered


def test_search_architecture_physics_aware_uses_physics_residual(poisson_problem):
    """physics_aware=True must actually incorporate physics_residual into
    the score, not just carry it along for inspection.

    Verified directly from a single physics_aware=True run's per-trial
    records (each trial reports its plain "accuracy" component alongside
    the combined "score" and "physics_residual" precisely so this can be
    checked without needing bit-identical reruns — model weight init is
    not seeded per trial, so two separate `search_architecture` calls are
    not guaranteed to reproduce identical scores even with matching
    architecture/hyperparameter sequences)."""
    space = _tiny_search_space()

    result_pa = search_architecture(
        poisson_problem, search_space=space, n_trials=4,
        epochs_per_trial=8, n_train=32, n_bc=16, n_col=48, grid_n=8,
        physics_aware=True, lam=1.0, seed=1,
    )

    assert np.isfinite(result_pa.best_score)
    scored_pa = [t for t in result_pa.all_trials if t.get("score") is not None]
    assert scored_pa
    # every trial in this search space trains a PINN-family model, so a
    # residual should always have been captured and actually folded into
    # the score as accuracy + lambda * physics_residual.
    for t in scored_pa:
        assert t["physics_residual"] is not None
        assert np.isfinite(t["physics_residual"])
        expected = t["accuracy"] + 1.0 * t["physics_residual"]
        assert t["score"] == pytest.approx(expected, rel=1e-6, abs=1e-9)
        # and the penalty must be nonzero-effective for this to be a real test
        assert t["score"] != pytest.approx(t["accuracy"], rel=1e-9)


def test_search_architecture_rejects_empty_search_space(poisson_problem):
    with pytest.raises(ValueError):
        search_architecture(poisson_problem, search_space=ArchitectureSearchSpace(candidates={}),
                            n_trials=1)


def test_nas_exported_from_pinneapple_arena():
    import pinneapple_arena
    assert hasattr(pinneapple_arena, "search_architecture")
    assert hasattr(pinneapple_arena, "ArchitectureSearchSpace")
    assert hasattr(pinneapple_arena, "NASResult")
