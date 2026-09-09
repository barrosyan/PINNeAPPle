"""Tests for pinneapple_analysis.trust.TrustGate."""
from __future__ import annotations

import torch

from pinneapple_analysis import TrustGate
from pinneapple_systems.component_library import ComponentRegistry
from pinneapple_systems.component_modeling.physics_residuals import heat_conduction_residual


def test_fit_and_ood_score_in_distribution_vs_far_out():
    gate = TrustGate()
    training_coords = torch.rand(200, 2)  # roughly Uniform[0,1]^2
    gate.fit(training_coords)

    in_dist = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    far_out = gate.score(model=None, x=torch.tensor([[50.0, 50.0]]))

    assert 0.0 <= in_dist.ood_score <= 1.0
    assert 0.0 <= far_out.ood_score <= 1.0
    assert in_dist.ood_score > far_out.ood_score


def test_fit_from_bounds():
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    result = gate.score(model=None, x=torch.tensor([[0.5, 0.5]]))
    assert result.ood_score is not None


def test_score_without_any_evidence_raises():
    gate = TrustGate()
    try:
        gate.score(model=None, x=torch.rand(1, 2))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_residual_score_low_for_trained_model_high_error_for_random():
    model = ComponentRegistry.build("HeatExchangerVanillaPINN", in_dim=2, out_dim=1, hidden=(16, 16))
    gate = TrustGate()
    gate.fit_from_bounds({"x": (0.0, 1.0), "y": (0.0, 1.0)})
    x = torch.rand(16, 2)
    result = gate.score(model=model, x=x, residual_fn=heat_conduction_residual, residual_kwargs={"k": 1.0, "Q": 0.0})
    assert result.residual_score is not None
    assert 0.0 < result.combined <= 1.0


def test_ensemble_score():
    class FakeEnsemble:
        def predict(self, x):
            mean = torch.full((x.shape[0], 1), 10.0)
            std = torch.full((x.shape[0], 1), 0.01)  # 0.1% relative std -> high confidence
            return mean, std

    gate = TrustGate()
    result = gate.score(model=None, x=torch.rand(4, 2), ensemble=FakeEnsemble())
    assert result.ensemble_score is not None
    assert result.ensemble_score > 0.9  # tiny std -> high confidence
