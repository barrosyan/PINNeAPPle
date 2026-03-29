"""pinneaple_uq — Uncertainty Quantification for physics-informed neural networks.

This module provides a unified, PyTorch-native toolkit for estimating and
evaluating predictive uncertainty in PINNs and related surrogate models.

Supported UQ methods
--------------------
``MCDropout`` / ``MCDropoutWrapper``
    Monte Carlo Dropout: runs *n* stochastic forward passes at inference time
    to approximate the posterior predictive distribution.

``EnsembleUQ``
    Deep Ensembles: combines predictions from multiple independently-trained
    models to estimate epistemic uncertainty.

``ConformalPredictor``
    Split Conformal Prediction: provides distribution-free, finite-sample
    coverage guarantees using a held-out calibration set.

Calibration utilities
---------------------
``CalibrationMetrics``
    Static methods for ECE, coverage, sharpness, and Gaussian NLL.

Unified interface
-----------------
``uq_predict(model, x, method="mc_dropout", **kwargs) -> UQResult``
    Single entry-point for uncertainty-aware prediction regardless of the
    underlying UQ method.

Quick start
-----------
>>> from pinneaple_uq import uq_predict, ConformalPredictor, CalibrationMetrics
>>> result = uq_predict(model, x_test, method="mc_dropout", n_samples=200)
>>> lower, upper = result.confidence_interval(alpha=0.05)
>>> ece = CalibrationMetrics.expected_calibration_error(
...     result.mean, y_test, result.std
... )
"""
from __future__ import annotations

from pinneaple_uq.core import UQResult, uq_predict
from pinneaple_uq.mc_dropout import MCDropout, MCDropoutConfig, MCDropoutWrapper
from pinneaple_uq.ensemble import EnsembleConfig, EnsembleUQ
from pinneaple_uq.conformal import ConformalPredictor
from pinneaple_uq.calibration import CalibrationMetrics

__all__ = [
    # Core types and unified interface
    "UQResult",
    "uq_predict",
    # Monte Carlo Dropout
    "MCDropout",
    "MCDropoutConfig",
    "MCDropoutWrapper",
    # Ensemble UQ
    "EnsembleUQ",
    "EnsembleConfig",
    # Conformal prediction
    "ConformalPredictor",
    # Calibration
    "CalibrationMetrics",
]
