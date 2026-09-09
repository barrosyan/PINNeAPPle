"""Regression test: uq_predict(..., method="mc_dropout") must work with models
that return a ``ModelOutput``/``PINNOutput`` wrapper dataclass (e.g. SIREN,
VanillaPINN) instead of a plain tensor.

Prior to the fix, ``MCDropoutWrapper.predict_with_uncertainty`` called
``self.model(x)`` and immediately did ``.detach()`` on the result, assuming a
plain ``torch.Tensor``. SIREN (and other BaseModel/PINNBase subclasses) wrap
their prediction tensor in a small dataclass with a ``.y`` attribute, which
crashed with ``AttributeError: 'ModelOutput' object has no attribute 'detach'``.
"""
from __future__ import annotations

import torch

from pinneapple_neural.architectures.registry import ModelRegistry
from pinneapple_analysis.uncertainty import uq_predict


def test_uq_predict_mc_dropout_with_model_output_wrapper():
    """SIREN's forward() returns a ModelOutput wrapper; uq_predict must unwrap it."""
    torch.manual_seed(0)
    model = ModelRegistry.build("siren", in_dim=2, out_dim=1)
    x_test = torch.randn(10, 2)

    result = uq_predict(model, x_test, method="mc_dropout", n_samples=8)

    assert result.mean.shape == (10, 1)
    assert result.std.shape == (10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.std).all()
    assert (result.std >= 0).all()


def test_uq_predict_mc_dropout_with_plain_tensor_architecture():
    """VanillaPINN also wraps its output (PINNOutput); confirm it still works
    and that the non-wrapped-tensor code path is unaffected by the fix."""
    torch.manual_seed(0)
    model = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1)
    x_test = torch.randn(10, 2)

    result = uq_predict(model, x_test, method="mc_dropout", n_samples=8)

    assert result.mean.shape == (10, 1)
    assert result.std.shape == (10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.std).all()


def test_uq_predict_mc_dropout_with_plain_nn_module():
    """A plain nn.Module (no wrapper output) must still work after the fix."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )
    x_test = torch.randn(12, 3)

    result = uq_predict(model, x_test, method="mc_dropout", n_samples=8)

    assert result.mean.shape == (12, 1)
    assert result.std.shape == (12, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.std).all()
