"""pinneapple_analysis.trust.trust_gate — ``TrustGate``: pre-consumption
trust scoring of a trained model's own prediction, before it is allowed
to feed a live digital twin.

Distinct from two existing, adjacent mechanisms — this is neither of them:

- ``pinneapple_llm.guardrail.PhysicsGuardrail`` verifies a *problem spec*
  an LLM proposed (residual re-check, dimensional analysis, conservation)
  before training even starts.
- ``pinneapple_systems.digital_twin.monitoring.anomaly.MahalanobisDetector``
  scores a live sensor *innovation vector* against a running Kalman
  filter's state covariance, continuously, once a twin is already
  running.

``TrustGate`` scores one specific prediction from an already-trained
model, combining up to three independent [0, 1] sub-scores:

- an out-of-distribution (OOD) score: Mahalanobis distance of the query
  coordinates to the model's training-coordinate distribution, squashed
  via ``scipy.stats.chi2.sf`` — the same Mahalanobis-distance +
  chi-squared concept ``MahalanobisDetector`` already uses, reimplemented
  here (rather than called directly) because it fits once from a static
  training sample instead of updating per Kalman step, a different call
  shape;
- a residual score: re-evaluating a PDE residual on the model's own
  prediction (any ``pinneapple_systems.component_modeling
  .physics_residuals``-shaped ``residual(model, coords, **kwargs) ->
  Tensor`` function, or a ``pinneapple_systems.component_library.Physics``
  instance);
- an optional ensemble-variance score, if a ``DeepEnsemble`` (or anything
  exposing ``.predict(x) -> (mean, std)``) is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import chi2


def _unwrap(out: Any) -> torch.Tensor:
    return out.y if hasattr(out, "y") else out


@dataclass
class TrustScore:
    combined: float
    ood_score: Optional[float] = None
    residual_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class TrustGate:
    def __init__(self, *, ood_weight: float = 0.4, residual_weight: float = 0.4, ensemble_weight: float = 0.2):
        self.ood_weight = ood_weight
        self.residual_weight = residual_weight
        self.ensemble_weight = ensemble_weight
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._dim: Optional[int] = None

    # -- fitting the OOD reference distribution --------------------------

    def fit(self, training_coords: torch.Tensor) -> "TrustGate":
        """Fit the OOD reference from a real sample of training
        coordinates (mean + covariance)."""
        x = training_coords.detach().cpu().numpy().astype(np.float64)
        self._mean = x.mean(axis=0)
        cov = np.atleast_2d(np.cov(x, rowvar=False))
        self._cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-8)
        self._dim = x.shape[1]
        return self

    def fit_from_bounds(self, domain_bounds: Dict[str, Tuple[float, float]]) -> "TrustGate":
        """Cheap approximation when no real training-coordinate sample is
        kept around: treats the domain as a uniform box and derives a
        diagonal covariance from each axis's range (``var = range**2 / 12``,
        the variance of a ``Uniform(a, b)``). Coarser than :meth:`fit` —
        use it only when the real sample isn't available."""
        bounds = list(domain_bounds.values())
        mean = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=np.float64)
        var = np.array([((hi - lo) ** 2) / 12.0 for lo, hi in bounds], dtype=np.float64)
        self._mean = mean
        self._cov_inv = np.diag(1.0 / np.clip(var, 1e-8, None))
        self._dim = len(bounds)
        return self

    @staticmethod
    def coord_bounds_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Reads ``checkpoint["metadata"]["domain_bounds"]`` if present —
        the common place a training pipeline in this repo records domain
        bounds. Raises ``KeyError`` (loudly, not a silent empty dict) if
        absent, since a made-up domain would make :meth:`fit_from_bounds`
        silently meaningless."""
        meta = checkpoint.get("metadata", {}) or {}
        bounds = meta.get("domain_bounds")
        if not bounds:
            raise KeyError("checkpoint metadata has no 'domain_bounds' entry.")
        return {k: tuple(v) for k, v in bounds.items()}

    # -- sub-scores --------------------------------------------------------

    def _ood_score(self, x: torch.Tensor) -> float:
        if self._mean is None or self._cov_inv is None:
            raise RuntimeError("TrustGate.fit()/.fit_from_bounds() must be called before OOD scoring.")
        xq = x.detach().cpu().numpy().astype(np.float64)
        xq = xq.mean(axis=0) if xq.ndim == 2 else xq.reshape(-1)
        delta = xq - self._mean
        d2 = float(delta @ self._cov_inv @ delta)
        # Probability of a Mahalanobis distance this large or larger under
        # the fitted distribution — low p-value => far out of distribution
        # => low trust. 1.0 = perfectly in-distribution.
        return float(chi2.sf(d2, df=self._dim))

    def _residual_score(self, model: Any, x: torch.Tensor, residual_fn: Callable[..., torch.Tensor], residual_kwargs: Dict[str, Any]) -> float:
        r = residual_fn(model, x, **residual_kwargs)
        rmse = float(torch.sqrt((r ** 2).mean()).item())
        return 1.0 / (1.0 + rmse)

    def _ensemble_score(self, ensemble: Any, x: torch.Tensor) -> float:
        mean, std = ensemble.predict(x)
        mean_t = _unwrap(mean)
        rel_std = float((std.abs() / mean_t.abs().clamp_min(1e-6)).mean().item())
        return 1.0 / (1.0 + rel_std)

    # -- combined score ------------------------------------------------

    def score(
        self,
        model: Any,
        x: torch.Tensor,
        *,
        residual_fn: Optional[Callable[..., torch.Tensor]] = None,
        residual_kwargs: Optional[Dict[str, Any]] = None,
        ensemble: Optional[Any] = None,
    ) -> TrustScore:
        """Combines whichever sub-scores are available into one [0, 1]
        weighted average (renormalized over only the parts actually
        computed). Raises if none of {fitted OOD, residual_fn, ensemble}
        were provided — a trust score with no evidence is not a score."""
        details: Dict[str, Any] = {}
        parts, weights = [], []

        ood = None
        if self._mean is not None:
            ood = self._ood_score(x)
            details["ood_p_value"] = ood
            parts.append(ood)
            weights.append(self.ood_weight)

        residual = None
        if residual_fn is not None:
            residual = self._residual_score(model, x, residual_fn, residual_kwargs or {})
            details["residual_score"] = residual
            parts.append(residual)
            weights.append(self.residual_weight)

        ens = None
        if ensemble is not None:
            ens = self._ensemble_score(ensemble, x)
            details["ensemble_score"] = ens
            parts.append(ens)
            weights.append(self.ensemble_weight)

        if not parts:
            raise ValueError(
                "TrustGate.score() has no evidence to combine — call fit()/fit_from_bounds() first, "
                "and/or pass residual_fn and/or ensemble."
            )

        wsum = sum(weights)
        combined = sum(p * w for p, w in zip(parts, weights)) / wsum
        return TrustScore(combined=combined, ood_score=ood, residual_score=residual, ensemble_score=ens, details=details)
