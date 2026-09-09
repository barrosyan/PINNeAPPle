"""Tests for pinneapple_tools.surrogate_policy.SurrogatePolicy."""
from __future__ import annotations

import tempfile
import time
import os

from pinneapple_tools.surrogate_policy import SurrogatePolicy


def _policy() -> SurrogatePolicy:
    return SurrogatePolicy(os.path.join(tempfile.mkdtemp(), "cache.json"), slow_threshold_s=0.05)


def test_no_direct_solver_always_needs_model():
    policy = _policy()
    result = policy.classify("some_problem", has_direct_solver=False)
    assert result.needs_model is True
    assert result.reason == "no_direct_solver"


def test_assumed_expensive_geometry_needs_model_without_timing():
    policy = _policy()
    result = policy.classify("geo_problem", assumed_expensive_geometry=True)
    assert result.needs_model is True
    assert result.reason == "assumed_expensive_geometry"
    assert result.measured_s is None


def test_measured_fast_vs_slow():
    policy = _policy()

    fast = policy.classify("fast_problem", run_once=lambda: None)
    assert fast.needs_model is False
    assert fast.reason == "measured_fast"

    slow = policy.classify("slow_problem", run_once=lambda: time.sleep(0.1))
    assert slow.needs_model is True
    assert slow.reason == "measured_slow"


def test_measurement_is_cached():
    policy = _policy()
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1

    policy.classify("cached_problem", run_once=run_once)
    policy.classify("cached_problem", run_once=run_once)
    assert calls["n"] == 1


def test_recommend_architecture():
    policy = _policy()
    result = policy.classify("no_solver", has_direct_solver=False)
    assert policy.recommend_architecture(result, n_observations=0) == "vanilla_pinn"
    assert policy.recommend_architecture(result, n_observations=100) == "modified_mlp"

    fast_result = policy.classify("fast2", run_once=lambda: None)
    assert policy.recommend_architecture(fast_result) is None
