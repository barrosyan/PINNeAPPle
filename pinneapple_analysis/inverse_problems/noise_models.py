"""Noise models / data-misfit functions for inverse problems.

A *noise model* defines the negative log-likelihood of the observations given
the model predictions and unknown parameters.  Choosing an appropriate noise
model is critical for robust parameter identification.

Available models
----------------
GaussianMisfit          ‖G(θ) − y‖²_{Γ⁻¹}           standard; sensitive to outliers
HuberMisfit             hybrid L2/L1                  robust to moderate outliers
CauchyMisfit            log(1 + r²/γ²)                very robust; heavy-tailed
StudentTMisfit          Student-t negative log-likelihood; ν controls robustness
HeteroscedasticMisfit   per-sample noise variance σᵢ²  learned or known

All classes inherit from ``DataMisfitBase`` and return a scalar PyTorch tensor.

References
----------
- Durbin & Koopman (2001) "Time Series Analysis by State Space Methods"
- Huber (1964) "Robust Estimation of a Location Parameter"
- Zhu & Spall (2002) "A Modified Quasi-Newton Method for Inverse Problems"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class DataMisfitBase(ABC):
    """Abstract data-misfit / negative log-likelihood term."""

    @abstractmethod
    def __call__(
        self,
        predicted: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar misfit.

        Parameters
        ----------
        predicted : torch.Tensor, shape (N, k) or (k,)
            Model output G(θ) at observation locations.
        observed : torch.Tensor, same shape as predicted
            Measurement vector y.

        Returns
        -------
        torch.Tensor
            Scalar misfit value.
        """

    def residuals(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Return element-wise residuals r = predicted − observed."""
        return predicted - observed


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian misfit  (standard MSE / Mahalanobis)
# ──────────────────────────────────────────────────────────────────────────────

class GaussianMisfit(DataMisfitBase):
    """Gaussian (L2) data misfit: D(G, y) = ‖G − y‖²_{Γ⁻¹}.

    Parameters
    ----------
    noise_cov : torch.Tensor, optional
        Observation noise covariance Γ (k×k).  If None, assumes σ²·I.
    noise_std : float
        Noise standard deviation σ (used when noise_cov is None).
    reduction : str
        "mean" or "sum" over observations.
    """

    def __init__(
        self,
        noise_cov: Optional[torch.Tensor] = None,
        noise_std: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        self.noise_cov = noise_cov
        self.noise_var = float(noise_std) ** 2
        self.reduction = reduction
        self._cov_inv: Optional[torch.Tensor] = None

    def _get_cov_inv(self, device) -> Optional[torch.Tensor]:
        if self.noise_cov is None:
            return None
        if self._cov_inv is None:
            self._cov_inv = torch.linalg.inv(self.noise_cov)
        return self._cov_inv.to(device)

    def __call__(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r = predicted - observed
        cov_inv = self._get_cov_inv(r.device)
        if cov_inv is not None:
            # Mahalanobis: r^T Γ⁻¹ r, averaged over batch
            if r.ndim == 1:
                misfit = r @ cov_inv @ r
            else:
                misfit = torch.sum((r @ cov_inv) * r, dim=-1).mean()
        else:
            misfit = torch.sum(r ** 2) / self.noise_var
            if self.reduction == "mean":
                misfit = misfit / max(r.numel(), 1)
        return misfit


# ──────────────────────────────────────────────────────────────────────────────
# Huber misfit  (robust to moderate outliers)
# ──────────────────────────────────────────────────────────────────────────────

class HuberMisfit(DataMisfitBase):
    """Huber (pseudo-Huber) data misfit.

    Behaves like L2 for |r| < δ and L1 for |r| > δ, giving robustness to
    moderate outliers while retaining smooth gradients.

    Pseudo-Huber (smooth variant): √(r² + δ²) − δ

    Parameters
    ----------
    delta : float
        Threshold between L2 and L1 regimes (same units as the residual).
    noise_std : float
        Scales residuals: r̃ = r / σ before applying the loss.
    smooth : bool
        If True use smooth pseudo-Huber; if False use standard (kink at ±δ).
    """

    def __init__(
        self,
        delta: float = 1.0,
        noise_std: float = 1.0,
        smooth: bool = True,
    ) -> None:
        self.delta = float(delta)
        self.noise_std = float(noise_std)
        self.smooth = smooth

    def __call__(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r = (predicted - observed) / self.noise_std
        if self.smooth:
            # Pseudo-Huber: δ²(√(1 + (r/δ)²) − 1)
            loss = self.delta ** 2 * (torch.sqrt(1.0 + (r / self.delta) ** 2) - 1.0)
        else:
            abs_r = torch.abs(r)
            loss = torch.where(
                abs_r <= self.delta,
                0.5 * r ** 2,
                self.delta * (abs_r - 0.5 * self.delta),
            )
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Cauchy misfit  (heavy-tailed, very robust)
# ──────────────────────────────────────────────────────────────────────────────

class CauchyMisfit(DataMisfitBase):
    """Cauchy / Lorentzian data misfit: log(1 + (r/γ)²).

    Heavy-tailed negative log-likelihood of the Cauchy distribution.
    Very robust to large outliers but may be harder to optimize.

    Parameters
    ----------
    gamma : float
        Scale parameter (half-width at half-maximum).
    noise_std : float
        Pre-normalisation of residuals.
    """

    def __init__(self, gamma: float = 1.0, noise_std: float = 1.0) -> None:
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)

    def __call__(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r = (predicted - observed) / self.noise_std
        return torch.log1p((r / self.gamma) ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Student-t misfit
# ──────────────────────────────────────────────────────────────────────────────

class StudentTMisfit(DataMisfitBase):
    """Student-t negative log-likelihood misfit.

    Negative log-likelihood of Student-t distribution with ν degrees of
    freedom.  As ν → ∞ converges to Gaussian; small ν → Cauchy.

    Parameters
    ----------
    nu : float
        Degrees of freedom ν > 2 (variance is finite for ν > 2).
    noise_std : float
        Scale parameter σ.
    """

    def __init__(self, nu: float = 5.0, noise_std: float = 1.0) -> None:
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.nu = float(nu)
        self.noise_std = float(noise_std)

    def __call__(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r = (predicted - observed) / self.noise_std
        # Negative log of Student-t density (up to constants):
        # ℓ(r) = ((ν+1)/2) log(1 + r²/ν)
        return ((self.nu + 1) / 2) * torch.log1p(r ** 2 / self.nu).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Heteroscedastic Gaussian misfit
# ──────────────────────────────────────────────────────────────────────────────

class HeteroscedasticMisfit(DataMisfitBase):
    """Gaussian misfit with per-observation noise variance.

    D(G, y) = Σᵢ [r²ᵢ / (2σᵢ²) + log σᵢ]

    This is the full negative log-likelihood, including the log-det term that
    penalises large predicted variances.

    Parameters
    ----------
    log_noise_var : torch.Tensor, shape (k,) or nn.Parameter
        Log-variance log(σᵢ²) for each observation dimension.  Can be a
        trainable ``nn.Parameter`` for noise-level learning.
    """

    def __init__(self, log_noise_var: torch.Tensor) -> None:
        self.log_noise_var = log_noise_var

    def __call__(self, predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        r = predicted - observed
        lv = self.log_noise_var.to(r.device)
        precision = torch.exp(-lv)
        nll = 0.5 * (r ** 2 * precision + lv)
        return nll.mean()


__all__ = [
    "DataMisfitBase",
    "GaussianMisfit",
    "HuberMisfit",
    "CauchyMisfit",
    "StudentTMisfit",
    "HeteroscedasticMisfit",
]
