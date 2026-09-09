"""Tests for pinneapple_tools.model_evaluation.metrics.axis_binned_profile."""
from __future__ import annotations

import numpy as np

from pinneapple_tools.model_evaluation.metrics import axis_binned_profile, regression_metrics


def test_scalar_field_binned_correctly():
    # coord in [0, 10), field = coord (perfect linear profile) -- each bin's
    # mean field should closely track its own mean coord.
    rng = np.random.default_rng(0)
    coord = rng.uniform(0, 10, size=10000)
    field = coord.copy()

    out = axis_binned_profile(coord, field, n_bins=20)
    assert out["bin_centers"].shape == (20,)
    assert out["bin_means"].shape == (20,)
    assert np.allclose(out["bin_centers"], out["bin_means"], atol=0.05)


def test_vector_field_preserves_trailing_shape():
    rng = np.random.default_rng(1)
    coord = rng.uniform(0, 2, size=5000)
    field = np.stack([coord, -coord, coord * 2], axis=1)  # (N, 3)

    out = axis_binned_profile(coord, field, n_bins=10)
    assert out["bin_means"].shape == (10, 3)
    assert np.allclose(out["bin_means"][:, 0], out["bin_centers"], atol=0.05)
    assert np.allclose(out["bin_means"][:, 1], -out["bin_centers"], atol=0.05)
    assert np.allclose(out["bin_means"][:, 2], 2 * out["bin_centers"], atol=0.05)


def test_bin_centers_are_actual_observed_mean_not_geometric_midpoint():
    # Concentrate points near the low end of each bin so the observed mean
    # differs measurably from the geometric bin midpoint.
    coord = np.array([0.0, 0.01, 0.02, 1.0, 1.01, 1.02])
    field = coord * 10
    out = axis_binned_profile(coord, field, n_bins=2)
    # bin 0 spans roughly [0, 0.51), its 3 points average to ~0.01
    assert out["bin_centers"][0] < 0.1
    assert abs(out["bin_centers"][0] - 0.01) < 1e-6


def test_empty_bin_gets_nan_mean_and_midpoint_center():
    coord = np.array([0.0, 0.0, 0.0, 9.0, 9.0])  # nothing lands in the middle bins
    field = coord.copy()
    out = axis_binned_profile(coord, field, n_bins=3)
    # middle bin (roughly [3, 6)) should be empty
    assert np.isnan(out["bin_means"][1])
    edges = np.linspace(0.0, 9.0, 4)
    assert np.isclose(out["bin_centers"][1], 0.5 * (edges[1] + edges[2]))


def test_composes_with_regression_metrics_for_profile_rmse():
    rng = np.random.default_rng(2)
    coord = rng.uniform(0, 5, size=2000)
    real_field = np.sin(coord)
    pred_field = np.sin(coord) + 0.01  # small constant offset "model error"

    real = axis_binned_profile(coord, real_field, n_bins=25)
    pred = axis_binned_profile(coord, pred_field, n_bins=25)

    metrics = regression_metrics(real["bin_means"], pred["bin_means"])
    assert metrics["overall"]["rmse"] < 0.05
    assert metrics["overall"]["rmse"] > 0.0
