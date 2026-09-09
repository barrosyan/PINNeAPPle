"""Regression test: AleatoricHead.forward and decompose_uncertainty must work
with models that return a ``ModelOutput``/``PINNOutput`` wrapper dataclass
(e.g. SIREN, VanillaPINN) instead of a plain tensor.

Prior to the fix:

* ``AleatoricHead.forward`` called ``self.base(x)`` and immediately did
  ``raw.ndim``, assuming a plain ``torch.Tensor``. SIREN (and other
  BaseModel/PINNBase subclasses) wrap their prediction tensor in a small
  dataclass with a ``.y`` attribute, which crashed with
  ``AttributeError: 'ModelOutput' object has no attribute 'ndim'``.

* ``decompose_uncertainty`` called ``model(x)`` and, for a model that doesn't
  return a ``(mean, log_var)`` tuple, appended the raw return value straight
  into its samples list before calling ``torch.stack`` on it. When the model
  returns a ``ModelOutput``/``PINNOutput`` wrapper instead of a plain tensor,
  this crashed with
  ``TypeError: expected Tensor as element 0 in argument 0, but got ModelOutput``.

Both were fixed by reusing this repo's existing duck-typed unwrap idiom
(``if hasattr(out, "y"): out = out.y``), matching the pattern already used in
``pinneapple_analysis/uncertainty/mc_dropout.py``.
"""
from __future__ import annotations

import torch

from pinneapple_neural.architectures.registry import ModelRegistry
from pinneapple_analysis.uncertainty.aleatoric import AleatoricHead
from pinneapple_analysis.uncertainty.decomposition import decompose_uncertainty


# ---------------------------------------------------------------------------
# AleatoricHead.forward
# ---------------------------------------------------------------------------

def test_aleatoric_head_forward_with_siren_model_output_wrapper():
    """SIREN's forward() returns a ModelOutput wrapper; AleatoricHead must unwrap it."""
    torch.manual_seed(0)
    base = ModelRegistry.build("siren", in_dim=2, out_dim=1)
    head = AleatoricHead(base, out_dim=1)
    x = torch.randn(10, 2)

    mean, log_var = head(x)

    assert mean.shape == (10, 1)
    assert log_var.shape == (10, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()


def test_aleatoric_head_forward_with_vanilla_pinn_pinn_output_wrapper():
    """VanillaPINN's forward() returns a PINNOutput wrapper; must also be unwrapped."""
    torch.manual_seed(0)
    base = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1)
    head = AleatoricHead(base, out_dim=1)
    x = torch.randn(10, 2)

    mean, log_var = head(x)

    assert mean.shape == (10, 1)
    assert log_var.shape == (10, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()


def test_aleatoric_head_predict_with_uncertainty_with_siren():
    """End-to-end predict_with_uncertainty must also work through the wrapper."""
    torch.manual_seed(0)
    base = ModelRegistry.build("siren", in_dim=2, out_dim=1)
    head = AleatoricHead(base, out_dim=1)
    x = torch.randn(10, 2)

    result = head.predict_with_uncertainty(x)

    assert result.mean.shape == (10, 1)
    assert result.aleatoric_std.shape == (10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.aleatoric_std).all()
    assert (result.aleatoric_std >= 0).all()
    assert (result.epistemic_std == 0).all()


# ---------------------------------------------------------------------------
# decompose_uncertainty
# ---------------------------------------------------------------------------

def test_decompose_uncertainty_with_siren_raw_model_epistemic_only():
    """decompose_uncertainty(has_aleatoric=False) called directly on a raw
    ModelOutput-wrapping model (no (mean, log_var) tuple) must unwrap ``.y``
    before stacking samples."""
    torch.manual_seed(0)
    model = ModelRegistry.build("siren", in_dim=2, out_dim=1)
    x = torch.randn(10, 2)

    result = decompose_uncertainty(model, x, n_samples=5, has_aleatoric=False)

    assert result.mean.shape == (10, 1)
    assert result.samples.shape == (5, 10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.epistemic_std).all()
    # aleatoric_var is zeros_like(...) then sqrt(var + eps), so it's a tiny
    # eps-scale residual (~sqrt(float32 eps) ~= 3.45e-4) rather than exactly zero.
    assert (result.aleatoric_std < 1e-3).all()


def test_decompose_uncertainty_with_vanilla_pinn_raw_model_epistemic_only():
    """Same as above but for a PINNOutput-wrapping model (VanillaPINN)."""
    torch.manual_seed(0)
    model = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1)
    x = torch.randn(10, 2)

    result = decompose_uncertainty(model, x, n_samples=5, has_aleatoric=False)

    assert result.mean.shape == (10, 1)
    assert result.samples.shape == (5, 10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.epistemic_std).all()


def test_decompose_uncertainty_via_aleatoric_head_and_mc_dropout_wrapper():
    """Recommended composed usage: AleatoricHead(siren) wrapped in
    MCDropoutWrapper, decomposed into aleatoric + epistemic parts. Exercises
    both fixed unwrap sites together (AleatoricHead.forward and
    decompose_uncertainty)."""
    from pinneapple_analysis.uncertainty.mc_dropout import MCDropoutConfig, MCDropoutWrapper

    torch.manual_seed(0)
    base = ModelRegistry.build("siren", in_dim=2, out_dim=1)
    head = AleatoricHead(base, out_dim=1)
    mcd = MCDropoutWrapper(head, MCDropoutConfig(n_samples=5))
    x = torch.randn(10, 2)

    result = decompose_uncertainty(mcd, x, n_samples=5, has_aleatoric=True)

    assert result.mean.shape == (10, 1)
    assert result.aleatoric_std.shape == (10, 1)
    assert result.epistemic_std.shape == (10, 1)
    assert torch.isfinite(result.mean).all()
    assert torch.isfinite(result.aleatoric_std).all()
    assert torch.isfinite(result.epistemic_std).all()
    assert (result.aleatoric_std >= 0).all()
    assert (result.epistemic_std >= 0).all()
