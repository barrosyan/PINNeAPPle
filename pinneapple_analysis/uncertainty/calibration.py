"""Calibration metrics for probabilistic regression models.

A well-calibrated model should assign uncertainty estimates that match
empirical frequencies.  This module provides standard metrics for diagnosing
and reporting calibration quality.

References
----------
Kuleshov, V., Fenner, N. & Ermon, S. (2018). *Accurate Uncertainties for Deep
Learning Using Calibrated Regression*. ICML.

Levi, D. et al. (2022). *Evaluating and Calibrating Uncertainty Prediction in
Regression Tasks*. Sensors, 22(15).
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import Tensor

# Optional scipy import for precise normal CDF.
try:
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def _normal_cdf(z: Tensor) -> Tensor:
        import numpy as np  # type: ignore
        return torch.from_numpy(_scipy_norm.cdf(z.cpu().numpy())).to(z.device)

    def _normal_ppf(p: float) -> float:
        return float(_scipy_norm.ppf(p))

    _HAS_SCIPY = True

except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

    def _normal_cdf(z: Tensor) -> Tensor:  # type: ignore[misc]
        """Approximation of the standard normal CDF using the error function."""
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    def _normal_ppf(p: float) -> float:  # type: ignore[misc]
        """Approximate inverse normal CDF via bisection (fallback)."""
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            cdf_mid = float(_normal_cdf(torch.tensor(mid)).item())
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0


class CalibrationMetrics:
    """Collection of calibration and sharpness metrics for regression UQ.

    All methods are static/class methods and accept plain ``Tensor`` inputs.
    No instantiation is required; use the class directly as a namespace:

    .. code-block:: python

        ece = CalibrationMetrics.expected_calibration_error(
            y_pred, y_true, y_std, n_bins=15
        )
        cov = CalibrationMetrics.coverage_at_level(y_pred, y_true, y_std, alpha=0.1)

    Notes
    -----
    All tensors should be 1-D or 2-D.  For multi-output models, metrics are
    computed element-wise and then averaged across output dimensions.
    """

    # ------------------------------------------------------------------
    # Expected Calibration Error
    # ------------------------------------------------------------------

    @staticmethod
    def expected_calibration_error(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute the Expected Calibration Error (ECE) for regression.

        For each confidence level ``p`` in a uniform grid over ``(0, 1)``,
        the empirical coverage is computed and compared to the nominal
        coverage.  ECE is the area-weighted average absolute difference.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        n_bins:
            Number of equally-spaced confidence levels at which to evaluate
            calibration.

        Returns
        -------
        float
            ECE in ``[0, 1]``.  A perfectly calibrated model returns 0.
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_std = y_std.float().clamp(min=1e-9)

        # Standardised residuals.
        z = (y_true - y_pred) / y_std

        levels = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1]  # exclude 0 and 1
        ece_sum = 0.0
        for p in levels.tolist():
            z_p = _normal_ppf(p)
            empirical = float((_normal_cdf(z) <= p).float().mean().item())
            ece_sum += abs(empirical - p)

        return ece_sum / n_bins

    # ------------------------------------------------------------------
    # Coverage at a given level
    # ------------------------------------------------------------------

    @staticmethod
    def coverage_at_level(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        alpha: float = 0.1,
    ) -> float:
        """Compute empirical coverage of the (1 - alpha) Gaussian interval.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        alpha:
            Miscoverage level; e.g. ``0.1`` targets 90 % coverage.

        Returns
        -------
        float
            Fraction of samples whose true value lies within the predicted
            interval.
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_std = y_std.float().clamp(min=1e-9)

        z_alpha = _normal_ppf(1.0 - alpha / 2.0)
        lower = y_pred - z_alpha * y_std
        upper = y_pred + z_alpha * y_std
        within = (y_true >= lower) & (y_true <= upper)

        if within.dim() > 1:
            within = within.all(dim=1)
        return float(within.float().mean().item())

    # ------------------------------------------------------------------
    # Calibration plot data
    # ------------------------------------------------------------------

    @staticmethod
    def calibration_plot_data(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
        n_bins: int = 15,
    ) -> Dict[str, List[float]]:
        """Compute data for a reliability / calibration diagram.

        Returns a dict with ``"expected"`` and ``"observed"`` confidence
        levels that can be passed directly to a plotting library.

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.
        n_bins:
            Number of confidence levels.

        Returns
        -------
        dict with keys:
            ``"expected"`` — list of nominal confidence levels.
            ``"observed"`` — list of empirical coverages at each level.
            ``"ece"`` — scalar ECE (float wrapped in a single-element list for
            consistency).
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_std = y_std.float().clamp(min=1e-9)

        z = (y_true - y_pred) / y_std
        levels = torch.linspace(0.0, 1.0, n_bins + 2)[1:-1].tolist()

        observed: List[float] = []
        for p in levels:
            empirical = float((_normal_cdf(z) <= p).float().mean().item())
            observed.append(empirical)

        ece = sum(abs(o - e) for o, e in zip(observed, levels)) / n_bins

        return {
            "expected": list(levels),
            "observed": observed,
            "ece": [ece],
        }

    # ------------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------------

    @staticmethod
    def sharpness(y_std: Tensor) -> float:
        """Compute sharpness: the mean predictive standard deviation.

        Lower sharpness indicates more confident (narrower) predictions.
        Sharpness should be interpreted alongside calibration — a model can
        be sharp but miscalibrated, or well-calibrated but not sharp.

        Parameters
        ----------
        y_std:
            Predictive standard deviation, shape ``(N,)`` or ``(N, D)``.

        Returns
        -------
        float
            Mean predictive std, averaged over all elements.
        """
        return float(y_std.float().mean().item())

    # ------------------------------------------------------------------
    # Gaussian NLL
    # ------------------------------------------------------------------

    @staticmethod
    def nll_gaussian(
        y_pred: Tensor,
        y_true: Tensor,
        y_std: Tensor,
    ) -> float:
        """Compute the mean Gaussian negative log-likelihood.

        Under the assumption that ``p(y | x) = N(y_pred, y_std²)``, the NLL
        for a single sample is:

        .. math::

            \\text{NLL} = \\frac{1}{2} \\log(2\\pi\\sigma^2)
                        + \\frac{(y - \\mu)^2}{2\\sigma^2}

        Parameters
        ----------
        y_pred:
            Predictive mean, shape ``(N,)`` or ``(N, D)``.
        y_true:
            Ground-truth targets, same shape as *y_pred*.
        y_std:
            Predictive standard deviation, same shape as *y_pred*.

        Returns
        -------
        float
            Mean NLL over all samples and output dimensions.
        """
        y_pred = y_pred.float()
        y_true = y_true.float()
        y_std = y_std.float().clamp(min=1e-9)

        var = y_std ** 2
        nll = 0.5 * (torch.log(2.0 * math.pi * var) + ((y_true - y_pred) ** 2) / var)
        return float(nll.mean().item())
