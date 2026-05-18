"""Regularization framework for inverse problems.

Provides differentiable regularization terms that penalise unphysical or
unlikely parameter values.  All regularizers accept PyTorch tensors so they
compose naturally with PINN physics losses.

Supported regularizers
----------------------
TikhonovRegularizer       R(θ) = λ ‖θ − θ₀‖²₂            (L2 / ridge)
SparsityRegularizer       R(θ) = λ ‖θ‖₁                  (L1 / LASSO)
TotalVariationRegularizer R(u) = λ TV(u)                  (anisotropic 1D/2D)
CompositeRegularizer      R(θ) = Σᵢ wᵢ Rᵢ(θ)
LCurveSelector            auto λ via L-curve + max curvature

References
----------
- Hansen (1992) "Analysis of Discrete Ill-Posed Problems by Means of the
  L-Curve"
- Vogel (2002) "Computational Methods for Inverse Problems"
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class RegularizerBase(ABC):
    """Abstract regularization term R(θ) → scalar loss."""

    @abstractmethod
    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the regularization penalty.

        Parameters
        ----------
        theta : torch.Tensor, shape (p,) or (B, p)
            Parameter vector(s).

        Returns
        -------
        torch.Tensor
            Scalar penalty.
        """

    def weight(self, lam: float) -> "CompositeRegularizer":
        """Shorthand: scale this regularizer and return a composite."""
        return CompositeRegularizer([(lam, self)])


# ──────────────────────────────────────────────────────────────────────────────
# Tikhonov (L2)
# ──────────────────────────────────────────────────────────────────────────────

class TikhonovRegularizer(RegularizerBase):
    """Tikhonov / ridge regularizer: R(θ) = λ ‖θ − θ₀‖²₂.

    Parameters
    ----------
    lambda_reg : float
        Regularization strength λ > 0.
    prior : torch.Tensor or None
        Prior mean θ₀.  Defaults to **zero** if None.
    covariance : torch.Tensor or None
        Prior covariance C₀ (p×p).  If given, penalty becomes
        (θ−θ₀)^T C₀⁻¹ (θ−θ₀) (Mahalanobis distance).
    """

    def __init__(
        self,
        lambda_reg: float,
        prior: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
    ) -> None:
        self.lambda_reg = float(lambda_reg)
        self.prior = prior
        self.covariance = covariance
        self._cov_inv: Optional[torch.Tensor] = None

    def _get_cov_inv(self, device) -> Optional[torch.Tensor]:
        if self.covariance is None:
            return None
        if self._cov_inv is None:
            self._cov_inv = torch.linalg.inv(self.covariance.to(device))
        return self._cov_inv.to(device)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        delta = theta - (self.prior.to(theta.device) if self.prior is not None else 0.0)
        cov_inv = self._get_cov_inv(theta.device)
        if cov_inv is not None:
            penalty = delta @ cov_inv @ delta
        else:
            penalty = torch.sum(delta ** 2)
        return self.lambda_reg * penalty


# ──────────────────────────────────────────────────────────────────────────────
# Sparsity (L1 / LASSO)
# ──────────────────────────────────────────────────────────────────────────────

class SparsityRegularizer(RegularizerBase):
    """L1 / LASSO regularizer: R(θ) = λ ‖θ‖₁.

    Promotes sparse parameter identification (many parameters near zero).
    Uses a differentiable approximation near zero to avoid gradient issues.

    Parameters
    ----------
    lambda_reg : float
        Regularization strength.
    eps : float
        Smoothing parameter for the L1 approximation `sqrt(θ² + ε²)`.
        Set to 0 for exact L1 (non-smooth at 0).
    """

    def __init__(self, lambda_reg: float, eps: float = 1e-6) -> None:
        self.lambda_reg = float(lambda_reg)
        self.eps = float(eps)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            penalty = torch.sum(torch.sqrt(theta ** 2 + self.eps ** 2))
        else:
            penalty = torch.sum(torch.abs(theta))
        return self.lambda_reg * penalty


# ──────────────────────────────────────────────────────────────────────────────
# Total Variation
# ──────────────────────────────────────────────────────────────────────────────

class TotalVariationRegularizer(RegularizerBase):
    """Anisotropic Total Variation regularizer for spatial field parameters.

    Supports:
    - 1D field: θ ∈ R^n, TV(θ) = Σᵢ |θᵢ₊₁ − θᵢ|
    - 2D field: θ ∈ R^{H×W}, TV(θ) = Σ |∂ₓθ| + |∂ᵧθ|

    Parameters
    ----------
    lambda_reg : float
    eps : float
        Smooth approximation: |x| ≈ sqrt(x² + ε²).
    """

    def __init__(self, lambda_reg: float, eps: float = 1e-6) -> None:
        self.lambda_reg = float(lambda_reg)
        self.eps = float(eps)

    def _smooth_abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x ** 2 + self.eps ** 2)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            diff = theta[1:] - theta[:-1]
            penalty = self._smooth_abs(diff).sum()
        elif theta.ndim == 2:
            dx = theta[:, 1:] - theta[:, :-1]
            dy = theta[1:, :] - theta[:-1, :]
            penalty = self._smooth_abs(dx).sum() + self._smooth_abs(dy).sum()
        elif theta.ndim == 3:
            dx = theta[:, :, 1:] - theta[:, :, :-1]
            dy = theta[:, 1:, :] - theta[:, :-1, :]
            dz = theta[1:, :, :] - theta[:-1, :, :]
            penalty = (
                self._smooth_abs(dx).sum()
                + self._smooth_abs(dy).sum()
                + self._smooth_abs(dz).sum()
            )
        else:
            raise ValueError(f"TotalVariationRegularizer: unsupported shape {theta.shape}")
        return self.lambda_reg * penalty


# ──────────────────────────────────────────────────────────────────────────────
# Composite
# ──────────────────────────────────────────────────────────────────────────────

class CompositeRegularizer(RegularizerBase):
    """Weighted sum of multiple regularizers: R(θ) = Σᵢ wᵢ Rᵢ(θ).

    Parameters
    ----------
    terms : list of (weight, RegularizerBase)
    """

    def __init__(self, terms: Optional[List[Tuple[float, RegularizerBase]]] = None) -> None:
        self.terms: List[Tuple[float, RegularizerBase]] = list(terms) if terms else []

    def add(self, weight: float, reg: RegularizerBase) -> "CompositeRegularizer":
        """Append a weighted regularizer in-place and return self."""
        self.terms.append((weight, reg))
        return self

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=theta.device, dtype=theta.dtype)
        for w, reg in self.terms:
            total = total + w * reg(theta)
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Regularization parameter selector: L-curve
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LCurveResult:
    """Result of an L-curve analysis.

    Attributes
    ----------
    lambdas : list of float
        Tested λ values.
    data_misfits : list of float
        Data misfit ‖G(θ_λ) − y‖ for each λ.
    regularization_norms : list of float
        ‖θ_λ − θ₀‖ for each λ.
    optimal_lambda : float
        λ at the corner of the L-curve (maximum curvature point).
    curvatures : list of float
        Discrete curvature estimate for each λ.
    """
    lambdas: List[float]
    data_misfits: List[float]
    regularization_norms: List[float]
    optimal_lambda: float
    curvatures: List[float]


class LCurveSelector:
    """Automatic regularization parameter selection via the L-curve.

    The L-curve plots ‖residual‖ vs ‖regularization norm‖ (log-log scale) for
    a range of λ values.  The optimal λ is at the corner — the point of maximum
    discrete curvature.

    Parameters
    ----------
    train_fn : callable(lambda_reg: float) -> (data_misfit: float, reg_norm: float)
        Function that trains the inverse problem with the given λ and returns
        the two L-curve coordinates.
    lambdas : list of float, optional
        λ grid to search.  Defaults to 30 log-spaced values in [1e-6, 1e2].

    Example
    -------
    >>> selector = LCurveSelector(lambda lam: train_and_eval(lam))
    >>> result = selector.select()
    >>> print(f"Optimal λ = {result.optimal_lambda:.2e}")
    """

    def __init__(
        self,
        train_fn: Callable[[float], Tuple[float, float]],
        lambdas: Optional[List[float]] = None,
    ) -> None:
        self.train_fn = train_fn
        self.lambdas = lambdas or list(np.logspace(-6, 2, 30))

    def select(self, verbose: bool = True) -> LCurveResult:
        """Run the L-curve sweep and return the result."""
        data_misfits: List[float] = []
        reg_norms: List[float] = []

        for i, lam in enumerate(self.lambdas):
            dm, rn = self.train_fn(float(lam))
            data_misfits.append(float(dm))
            reg_norms.append(float(rn))
            if verbose:
                print(f"  L-curve [{i+1:3d}/{len(self.lambdas)}] λ={lam:.2e}  "
                      f"‖r‖={dm:.3e}  ‖θ‖={rn:.3e}")

        # Compute discrete curvature in log-log space
        log_dm = np.log(np.array(data_misfits) + 1e-30)
        log_rn = np.log(np.array(reg_norms) + 1e-30)
        curvatures = self._discrete_curvature(log_dm, log_rn)

        opt_idx = int(np.argmax(curvatures))
        optimal_lambda = float(self.lambdas[opt_idx])

        if verbose:
            print(f"  => Optimal λ = {optimal_lambda:.3e} (index {opt_idx})")

        return LCurveResult(
            lambdas=list(self.lambdas),
            data_misfits=data_misfits,
            regularization_norms=reg_norms,
            optimal_lambda=optimal_lambda,
            curvatures=list(curvatures),
        )

    @staticmethod
    def _discrete_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Discrete curvature of curve (x, y) using second-order finite differences."""
        n = len(x)
        kappa = np.zeros(n)
        for i in range(1, n - 1):
            dx1 = x[i] - x[i - 1]
            dx2 = x[i + 1] - x[i]
            dy1 = y[i] - y[i - 1]
            dy2 = y[i + 1] - y[i]
            # Menger curvature (triangle-based)
            dxdx = (dx1 + dx2) / 2
            dxdy = (dy1 / (dx1 + 1e-30) - dy2 / (dx2 + 1e-30)) / (dxdx + 1e-30)
            kappa[i] = abs(dxdy) / (1 + ((dy1 + dy2) / 2) ** 2) ** 1.5
        return kappa


__all__ = [
    "RegularizerBase",
    "TikhonovRegularizer",
    "SparsityRegularizer",
    "TotalVariationRegularizer",
    "CompositeRegularizer",
    "LCurveSelector",
    "LCurveResult",
]
