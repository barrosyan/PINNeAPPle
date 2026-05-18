"""Split Conformal Prediction for regression.

Split conformal prediction provides *distribution-free*, *finite-sample* valid
coverage guarantees without making any assumptions about the data-generating
process.

References
----------
Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a
Random World*. Springer.

Angelopoulos, A. N. & Bates, S. (2022). *A Gentle Introduction to Conformal
Prediction and Distribution-Free Uncertainty Quantification*.
arXiv:2107.07511.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor

# Scipy is used for exact quantiles; pure-PyTorch fallback is provided.
try:
    import numpy as np  # type: ignore
    from scipy.stats import rankdata as _rankdata  # type: ignore

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _conformal_quantile(scores: Tensor, alpha: float) -> float:
    """Compute the (1 - alpha) * (1 + 1/n) quantile of *scores*.

    This is the finite-sample corrected conformal quantile recommended by
    Angelopoulos & Bates (2022), Eq. (3).

    Parameters
    ----------
    scores:
        Non-conformity scores, shape ``(n_cal,)``.
    alpha:
        Miscoverage level; e.g. ``0.1`` targets 90 % coverage.

    Returns
    -------
    float
        Calibrated quantile value (the prediction interval half-width).
    """
    n = scores.numel()
    # Level with finite-sample correction.
    level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)

    if _HAS_SCIPY:
        q = float(np.quantile(scores.cpu().numpy(), level))
    else:
        # Pure-PyTorch: use torch.quantile (available since 1.7).
        q = float(torch.quantile(scores.float(), level))
    return q


class ConformalPredictor:
    """Split conformal predictor for regression tasks.

    Given a calibration set ``(x_cal, y_cal)`` and a coverage level
    ``1 - alpha``, :meth:`calibrate` computes the smallest prediction-interval
    half-width that achieves the target marginal coverage.

    :meth:`predict` then returns symmetric prediction intervals
    ``[y_hat - q, y_hat + q]`` for new inputs.

    Parameters
    ----------
    model:
        Any callable ``f: Tensor -> Tensor`` that returns point predictions.
        Typically a trained ``nn.Module``.
    alpha:
        Miscoverage level.  The predictor targets ``(1 - alpha)`` marginal
        coverage.  Must be in ``(0, 1)``.

    Raises
    ------
    ValueError
        If *alpha* is not in ``(0, 1)``.

    Examples
    --------
    >>> cp = ConformalPredictor(model, alpha=0.1)
    >>> cp.calibrate(x_cal, y_cal)
    >>> y_pred, lower, upper = cp.predict(x_test)
    >>> cov = cp.coverage(x_test, y_test)
    >>> print(f"Empirical coverage: {cov:.3f}")  # should be ≈ 0.90
    """

    def __init__(self, model: Callable[[Tensor], Tensor], alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha must be in (0, 1); got {alpha}."
            )
        self.model = model
        self.alpha = alpha
        self._quantile: Optional[float] = None
        self._n_cal: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """``True`` after :meth:`calibrate` has been called successfully."""
        return self._quantile is not None

    @property
    def quantile(self) -> float:
        """Calibrated prediction-interval half-width.

        Raises
        ------
        RuntimeError
            If the predictor has not been calibrated yet.
        """
        if self._quantile is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated. "
                "Call calibrate(x_cal, y_cal) first."
            )
        return self._quantile

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(self, x_cal: Tensor, y_cal: Tensor) -> None:
        """Compute non-conformity scores on the calibration set.

        Stores the ``(1 - alpha)`` quantile of the absolute residuals as the
        prediction-interval half-width for future calls to :meth:`predict`.

        Parameters
        ----------
        x_cal:
            Calibration inputs, shape ``(n_cal, ...)``.
        y_cal:
            Calibration targets, shape ``(n_cal, ...)`` or ``(n_cal,)``.
            Must be broadcastable against the model's output shape.

        Notes
        -----
        The non-conformity score used here is the per-sample L∞ (max absolute)
        residual across output dimensions, which yields a single scalar score
        per sample regardless of output dimensionality.
        """
        device = x_cal.device
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to(device)

        y_hat = self.model(x_cal).detach()
        y_cal_d = y_cal.to(device).detach()

        # Flatten output dims and take the max absolute residual per sample.
        residuals = (y_hat - y_cal_d).abs()
        if residuals.dim() > 1:
            scores = residuals.flatten(start_dim=1).max(dim=1).values
        else:
            scores = residuals

        self._quantile = _conformal_quantile(scores, self.alpha)
        self._n_cal = x_cal.shape[0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return point prediction and calibrated prediction intervals.

        Parameters
        ----------
        x:
            Test inputs, shape ``(N, ...)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(y_pred, lower, upper)`` where each tensor has the same shape as
            the model output for *x*.

        Raises
        ------
        RuntimeError
            If the predictor has not been calibrated.
        """
        q = self.quantile  # Raises if not calibrated.

        device = x.device
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to(device)

        y_pred = self.model(x).detach()
        q_tensor = torch.tensor(q, dtype=y_pred.dtype, device=device)
        lower = y_pred - q_tensor
        upper = y_pred + q_tensor
        return y_pred, lower, upper

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def coverage(self, x_test: Tensor, y_test: Tensor) -> float:
        """Compute empirical marginal coverage on a held-out test set.

        Coverage is defined as the fraction of test samples for which the true
        target falls within the prediction interval on **all** output
        dimensions simultaneously.

        Parameters
        ----------
        x_test:
            Test inputs, shape ``(N, ...)``.
        y_test:
            Ground-truth targets, shape ``(N, ...)`` or ``(N,)``.

        Returns
        -------
        float
            Empirical coverage in ``[0, 1]``.
        """
        _y_pred, lower, upper = self.predict(x_test)
        y_test_d = y_test.to(lower.device).detach()

        # A sample is covered if y is in [lower, upper] for all dimensions.
        covered = (y_test_d >= lower) & (y_test_d <= upper)
        if covered.dim() > 1:
            # All output dims must be covered.
            covered = covered.all(dim=1)
        return float(covered.float().mean().item())

    def __repr__(self) -> str:  # pragma: no cover
        status = (
            f"quantile={self._quantile:.4g}, n_cal={self._n_cal}"
            if self.is_calibrated
            else "not calibrated"
        )
        return f"ConformalPredictor(alpha={self.alpha}, {status})"
