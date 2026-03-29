"""Shared types and unified interface for pinneaple_uq.

This module provides:
- ``UQResult``: structured container for predictions with uncertainty estimates.
- ``uq_predict``: unified dispatch function for all supported UQ methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor

# Precomputed z-scores for common confidence levels (two-sided).
# Scipy is used when available for exactness; we fall back to a lookup table.
try:
    from scipy.stats import norm as _scipy_norm  # type: ignore

    def _z_score(alpha: float) -> float:
        """Return the z-score for a two-sided (1-alpha) confidence interval."""
        return float(_scipy_norm.ppf(1.0 - alpha / 2.0))

except ImportError:  # pragma: no cover
    _ZSCORE_TABLE: Dict[float, float] = {
        0.001: 3.2905,
        0.005: 2.8070,
        0.010: 2.5758,
        0.025: 2.2414,
        0.050: 1.9600,
        0.100: 1.6449,
        0.200: 1.2816,
        0.320: 1.0000,
        0.500: 0.6745,
    }

    def _z_score(alpha: float) -> float:  # type: ignore[misc]
        """Return an approximate z-score via nearest-neighbour lookup table.

        Scipy is not installed; install it for exact quantiles.
        """
        if alpha in _ZSCORE_TABLE:
            return _ZSCORE_TABLE[alpha]
        # Nearest-neighbour fallback.
        closest = min(_ZSCORE_TABLE, key=lambda k: abs(k - alpha))
        return _ZSCORE_TABLE[closest]


@dataclass
class UQResult:
    """Container for predictions with associated uncertainty estimates.

    Attributes
    ----------
    mean:
        Point-estimate prediction tensor, shape ``(N, D)`` or ``(N,)``.
    std:
        Standard deviation (aleatoric + epistemic), same shape as ``mean``.
    samples:
        Optional raw samples used to compute statistics, shape ``(S, N, D)``
        where *S* is the number of stochastic forward passes or ensemble
        members. ``None`` when the method does not expose raw samples.
    metadata:
        Arbitrary key-value store for method-specific information (e.g. method
        name, dropout probability, number of ensemble members).
    """

    mean: Tensor
    std: Tensor
    samples: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------

    def lower_bound(self, alpha: float = 0.05) -> Tensor:
        """Return the lower bound of the (1 - alpha) Gaussian interval.

        Parameters
        ----------
        alpha:
            Significance level; e.g. ``0.05`` gives a 95 % interval.

        Returns
        -------
        Tensor
            ``mean - z * std``, same shape as ``mean``.
        """
        z = _z_score(alpha)
        return self.mean - z * self.std

    def upper_bound(self, alpha: float = 0.05) -> Tensor:
        """Return the upper bound of the (1 - alpha) Gaussian interval.

        Parameters
        ----------
        alpha:
            Significance level; e.g. ``0.05`` gives a 95 % interval.

        Returns
        -------
        Tensor
            ``mean + z * std``, same shape as ``mean``.
        """
        z = _z_score(alpha)
        return self.mean + z * self.std

    def confidence_interval(
        self, alpha: float = 0.05
    ) -> tuple[Tensor, Tensor]:
        """Return a symmetric Gaussian confidence interval.

        Parameters
        ----------
        alpha:
            Significance level; e.g. ``0.05`` gives a 95 % interval.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(lower, upper)`` tensors, each with the same shape as ``mean``.
        """
        z = _z_score(alpha)
        lower = self.mean - z * self.std
        upper = self.mean + z * self.std
        return lower, upper

    def __repr__(self) -> str:  # pragma: no cover
        shape = tuple(self.mean.shape)
        method = self.metadata.get("method", "unknown")
        return (
            f"UQResult(shape={shape}, method={method!r}, "
            f"mean_std={self.std.mean().item():.4g})"
        )


# ---------------------------------------------------------------------------
# Unified prediction interface
# ---------------------------------------------------------------------------

def uq_predict(
    model: Any,
    x: Tensor,
    method: Literal["mc_dropout", "ensemble"] = "mc_dropout",
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> UQResult:
    """Unified interface for uncertainty-aware prediction.

    Dispatches to the appropriate UQ wrapper based on *method* and returns a
    :class:`UQResult`.  The model is **not** modified in place; a wrapper is
    created temporarily.

    Parameters
    ----------
    model:
        A trained ``nn.Module`` (or any callable returning a ``Tensor``).
    x:
        Input tensor, shape ``(N, ...)``.
    method:
        UQ method to use.  Supported values:

        * ``"mc_dropout"`` — Monte Carlo Dropout via
          :class:`~pinneaple_uq.mc_dropout.MCDropoutWrapper`.
        * ``"ensemble"`` — Ensemble UQ via
          :class:`~pinneaple_uq.ensemble.EnsembleUQ`.  When using this method,
          pass ``models`` as a keyword argument containing a list of
          ``nn.Module`` instances.

    device:
        Device to run inference on.  Defaults to the device of ``x`` when
        ``x`` is already a tensor, otherwise ``"cpu"``.
    **kwargs:
        Additional keyword arguments forwarded to the underlying UQ class or
        its ``predict_with_uncertainty`` method.

        For ``mc_dropout``:
            - ``n_samples`` (int, default 100) — stochastic forward passes.
            - ``dropout_p`` (float, default 0.1) — dropout probability.
            - ``seed`` (Optional[int]) — RNG seed for reproducibility.

        For ``ensemble``:
            - ``models`` (List[nn.Module], **required**) — ensemble members.

    Returns
    -------
    UQResult
        Prediction mean, std, (optionally) samples, and metadata.

    Raises
    ------
    ValueError
        If an unsupported *method* is requested.

    Examples
    --------
    >>> result = uq_predict(my_model, x, method="mc_dropout", n_samples=200)
    >>> lower, upper = result.confidence_interval(alpha=0.05)
    """
    if device is None:
        device = x.device if isinstance(x, Tensor) else torch.device("cpu")

    if method == "mc_dropout":
        from pinneaple_uq.mc_dropout import MCDropoutConfig, MCDropoutWrapper

        n_samples = kwargs.pop("n_samples", 100)
        dropout_p = kwargs.pop("dropout_p", 0.1)
        seed = kwargs.pop("seed", None)
        cfg = MCDropoutConfig(n_samples=n_samples, dropout_p=dropout_p, seed=seed)
        wrapper = MCDropoutWrapper(model, cfg)
        return wrapper.predict_with_uncertainty(x, n_samples=n_samples, device=device)

    elif method == "ensemble":
        from pinneaple_uq.ensemble import EnsembleConfig, EnsembleUQ

        models = kwargs.pop("models", None)
        if models is None:
            raise ValueError(
                "uq_predict with method='ensemble' requires a 'models' kwarg "
                "containing a list of nn.Module instances."
            )
        cfg_kwargs = {k: kwargs.pop(k) for k in ("n_members", "diversity_reg", "seed") if k in kwargs}
        cfg = EnsembleConfig(**cfg_kwargs)
        ensemble = EnsembleUQ(models, cfg)
        return ensemble.predict_with_uncertainty(x, device=device)

    else:
        raise ValueError(
            f"Unknown UQ method: {method!r}. "
            "Supported methods: 'mc_dropout', 'ensemble'."
        )
