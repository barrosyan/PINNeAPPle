"""pinneapple_uq — Uncertainty Quantification for physics-informed neural networks.

This module provides a unified, PyTorch-native toolkit for estimating and
evaluating predictive uncertainty in PINNs and related surrogate models.

Uncertainty types
-----------------
``AleatoricHead`` / ``aleatoric_nll_loss``
    **Aleatoric (data) uncertainty** — irreducible noise in the data.
    The model learns both mean μ(x) and log-variance log σ²(x).
    Optimized via heteroscedastic Gaussian NLL.

``MCDropout`` / ``EnsembleUQ``
    **Epistemic (model) uncertainty** — reducible uncertainty from limited data
    or model capacity.  MC Dropout runs *n* stochastic forward passes;
    ensembles aggregate multiple independently-trained models.

``decompose_uncertainty``
    **Decomposition** — splits total variance into aleatoric + epistemic
    components from *n* stochastic passes of an AleatoricHead model.

``ConformalPredictor``
    **Conformal prediction** — distribution-free coverage guarantees via a
    held-out calibration set.

Quantile regression
-------------------
``QuantileHead`` / ``QuantileLoss``
    Learn conditional quantiles via pinball loss; also accessible from
    ``pinneapple_timeseries.uncertainty``.

Calibration utilities
---------------------
``CalibrationMetrics``
    Static methods for ECE, coverage, sharpness, and Gaussian NLL.

Unified interface
-----------------
``uq_predict(model, x, method=...) -> UQResult``
    Single entry-point for all methods.  Supported *method* values:
    ``"mc_dropout"``, ``"ensemble"``, ``"aleatoric"``, ``"decompose"``.

Quick start
-----------
>>> from pinneapple_analysis.uncertainty import uq_predict, AleatoricHead, decompose_uncertainty
>>> # Aleatoric-only
>>> head = AleatoricHead(base_model, out_dim=1)
>>> result = uq_predict(head, x_test, method="aleatoric")
>>> print(result.aleatoric_std)
>>>
>>> # Full decomposition (aleatoric + epistemic via MC Dropout)
>>> from pinneapple_analysis.uncertainty import MCDropoutWrapper, MCDropoutConfig
>>> mcd = MCDropoutWrapper(head, MCDropoutConfig(n_samples=50))
>>> result = decompose_uncertainty(mcd, x_test)
>>> print(result.aleatoric_std, result.epistemic_std)
"""
from __future__ import annotations

from pinneapple_analysis.uncertainty.core import UQResult, uq_predict
from pinneapple_analysis.uncertainty.mc_dropout import MCDropout, MCDropoutConfig, MCDropoutWrapper
from pinneapple_analysis.uncertainty.ensemble import EnsembleConfig, EnsembleUQ
from pinneapple_analysis.uncertainty.conformal import ConformalPredictor
from pinneapple_analysis.uncertainty.calibration import CalibrationMetrics
from pinneapple_analysis.uncertainty.aleatoric import AleatoricHead, aleatoric_nll_loss
from pinneapple_analysis.uncertainty.decomposition import decompose_uncertainty
from pinneapple_analysis.uncertainty.quantile import QuantileConfig, QuantileHead, QuantileLoss, pinball_loss_torch

__all__ = [
    # Core types and unified interface
    "UQResult",
    "uq_predict",
    # Aleatoric (data) uncertainty
    "AleatoricHead",
    "aleatoric_nll_loss",
    # Uncertainty decomposition
    "decompose_uncertainty",
    # Monte Carlo Dropout (epistemic)
    "MCDropout",
    "MCDropoutConfig",
    "MCDropoutWrapper",
    # Ensemble UQ (epistemic)
    "EnsembleUQ",
    "EnsembleConfig",
    # Conformal prediction
    "ConformalPredictor",
    # Calibration
    "CalibrationMetrics",
    # Quantile regression
    "QuantileConfig",
    "QuantileHead",
    "QuantileLoss",
    "pinball_loss_torch",
]
