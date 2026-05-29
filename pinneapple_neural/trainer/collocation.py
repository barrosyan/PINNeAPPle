"""Adaptive collocation point sampling for PINN training.

Standard PINN training uses a fixed set of collocation points drawn uniformly
from the domain.  In practice, the PDE residual is not uniform — it is high in
regions where the solution changes rapidly or where the network is still learning.

Residual-based Adaptive Refinement (RAR) periodically resamples the active
collocation set with a density biased toward high-residual regions:

    p_i = α · r_i² / Σ r_j²  +  (1-α) / N_pool

where r_i is the PDE residual magnitude at pool point i, α is ``residual_frac``,
and 1-α is the uniform component (prevents collapse onto a single point).

This technique can halve the number of epochs needed to reach a target accuracy
for problems with sharp gradients or complex boundaries.

References
----------
Lu et al. (2021). "DeepXDE: A deep learning library for solving differential
equations." SIAM Review 63(1). (adaptive resampling appendix)

Wu et al. (2023). "A comprehensive study of non-adaptive and residual-based
adaptive sampling for physics-informed neural networks." Comput. Methods Appl.
Mech. Engrg. 403.

Usage
-----
::

    from pinneapple_neural.trainer.collocation import (
        AdaptiveCollocationSampler, AdaptiveCollocationConfig,
    )

    # Build pool — any torch.Tensor of shape (N_pool, D)
    pool = sample_domain_points(n=40_000)

    cfg     = AdaptiveCollocationConfig(n_active=8000, resample_every=200)
    sampler = AdaptiveCollocationSampler(pool, cfg)

    for epoch in range(n_epochs):
        pts = sampler.active_points()          # (n_active, D)

        # Compute residual at active points
        res = pde_residual(model, pts)         # (n_active, ...)
        sampler.record_residuals(res)          # stores magnitudes for next resample

        # Trigger periodic resampling (uses last recorded residuals)
        resampled = sampler.step(epoch)
        if resampled:
            pts = sampler.active_points()      # fresh points
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveCollocationConfig:
    """Configuration for ``AdaptiveCollocationSampler``.

    Attributes
    ----------
    n_active        : number of active collocation points used per training step
    n_pool          : total number of candidate pool points
                      (should be much larger than n_active, e.g. 5×)
    resample_every  : resample the active set every this many epochs
    residual_frac   : fraction of n_active drawn with residual-weighted density;
                      (1 - residual_frac) is drawn uniformly — prevents collapse
    residual_batch  : sub-batch size when evaluating residuals on the pool
                      (to avoid OOM on large pools + expensive physics)
    replace         : whether to sample with replacement (default False)
    """
    n_active: int = 8000
    n_pool: int = 40_000
    resample_every: int = 200
    residual_frac: float = 0.7
    residual_batch: int = 2000
    replace: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveCollocationSampler:
    """Residual-based Adaptive Refinement (RAR) for PINN collocation points.

    Maintains a large static pool of candidate points and an active subset.
    The active subset is periodically resampled with a density biased toward
    regions of high PDE residual.

    Parameters
    ----------
    pool            : (N_pool, D) tensor of candidate domain points (any dtype)
    config          : AdaptiveCollocationConfig
    residual_fn     : optional callable ``(model, pts) -> Tensor`` that computes
                      per-point residual magnitudes.  If provided, resampling can
                      be triggered automatically via ``step(epoch, model)``.
                      If None, you must call ``record_residuals()`` manually.

    Usage (manual residual recording)
    ----------------------------------
    ::

        sampler = AdaptiveCollocationSampler(pool, cfg)
        for epoch in range(n_epochs):
            pts = sampler.active_points()
            res = pde_loss(model, pts)          # compute residuals
            res_magnitudes = res.detach().pow(2).sum(dim=-1)
            sampler.record_residuals(res_magnitudes)
            sampler.step(epoch)

    Usage (automatic resampling with residual_fn)
    ---------------------------------------------
    ::

        def my_residual_fn(model, pts):
            res = compute_pde_residual(model, pts)
            return res.pow(2).sum(dim=-1)          # (N,) magnitudes

        sampler = AdaptiveCollocationSampler(pool, cfg, residual_fn=my_residual_fn)
        for epoch in range(n_epochs):
            pts = sampler.active_points()
            ...
            sampler.step(epoch, model=model)        # auto-evaluates full pool
    """

    def __init__(
        self,
        pool: torch.Tensor,
        config: Optional[AdaptiveCollocationConfig] = None,
        residual_fn: Optional[Callable] = None,
    ) -> None:
        self.cfg = config or AdaptiveCollocationConfig()
        self.residual_fn = residual_fn

        self._pool: torch.Tensor = pool
        self._n_pool = len(pool)

        # Residual magnitudes — shape (N_pool,), float32 numpy array
        self._pool_residuals: Optional[np.ndarray] = None

        # Active indices into the pool
        self._active_idx: np.ndarray = np.random.choice(
            self._n_pool,
            size=min(self.cfg.n_active, self._n_pool),
            replace=self.cfg.replace,
        )

        self._epoch_last_resample: int = -1
        self._resample_count: int = 0
        self._history: List[Dict] = []   # log of resample events

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def active_points(self) -> torch.Tensor:
        """Return the current active collocation points as a tensor."""
        return self._pool[self._active_idx]

    def record_residuals(self, residuals: torch.Tensor) -> None:
        """Record PDE residual magnitudes for the *active* points.

        Call this once per training step after computing the PDE loss.
        The magnitudes are stored and used when ``step()`` triggers a resample.

        Parameters
        ----------
        residuals : (n_active,) tensor of non-negative residual magnitudes
                    (typically ``pde_residual.pow(2).sum(dim=-1).detach()``)
        """
        r = residuals.detach().cpu().float().numpy()
        if self._pool_residuals is None:
            self._pool_residuals = np.ones(self._n_pool, dtype=np.float32)
        self._pool_residuals[self._active_idx] = np.abs(r)

    def step(self, epoch: int, model: Optional[nn.Module] = None) -> bool:
        """Trigger resampling if the epoch condition is met.

        Parameters
        ----------
        epoch : current training epoch (0-indexed)
        model : if ``residual_fn`` was set and ``model`` is provided, the full
                pool residuals are evaluated automatically before resampling.

        Returns
        -------
        True if a resample was performed, False otherwise.
        """
        if epoch <= 0 or epoch % self.cfg.resample_every != 0:
            return False
        if epoch == self._epoch_last_resample:
            return False

        # Optionally evaluate full pool residuals via residual_fn
        if self.residual_fn is not None and model is not None:
            self._eval_full_pool(model)

        self._resample()
        self._epoch_last_resample = epoch
        return True

    def resample_now(self, model: Optional[nn.Module] = None) -> None:
        """Force an immediate resample regardless of epoch schedule."""
        if self.residual_fn is not None and model is not None:
            self._eval_full_pool(model)
        self._resample()

    def replace_pool(self, new_pool: torch.Tensor) -> None:
        """Replace the entire candidate pool with new points.

        Useful for curriculum-based strategies where the domain changes during
        training (e.g., expanding time windows or refinement regions).
        """
        self._pool = new_pool
        self._n_pool = len(new_pool)
        self._pool_residuals = None
        self._active_idx = np.random.choice(
            self._n_pool,
            size=min(self.cfg.n_active, self._n_pool),
            replace=self.cfg.replace,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resample(self) -> None:
        n_active = min(self.cfg.n_active, self._n_pool)
        α = self.cfg.residual_frac

        if self._pool_residuals is None or α <= 0.0:
            # Pure uniform
            idx = np.random.choice(self._n_pool, size=n_active, replace=self.cfg.replace)
        else:
            r = self._pool_residuals.copy()
            r_sum = r.sum()
            if r_sum < 1e-30:
                # Residuals all zero — fall back to uniform
                p = np.ones(self._n_pool, dtype=np.float64) / self._n_pool
            else:
                p_res = r / r_sum                             # residual density
                p_uni = np.ones(self._n_pool) / self._n_pool  # uniform density
                p = α * p_res + (1.0 - α) * p_uni
                p = p / p.sum()

            n_res = int(round(α * n_active))
            n_uni = n_active - n_res

            # Draw residual-weighted
            try:
                idx_res = np.random.choice(self._n_pool, size=n_res, replace=False, p=p)
            except ValueError:
                idx_res = np.random.choice(self._n_pool, size=n_res, replace=True, p=p)

            # Draw uniform (from complement if possible)
            remaining = np.setdiff1d(np.arange(self._n_pool), idx_res, assume_unique=True)
            if len(remaining) >= n_uni:
                idx_uni = np.random.choice(remaining, size=n_uni, replace=False)
            else:
                idx_uni = np.random.choice(self._n_pool, size=n_uni, replace=True)

            idx = np.concatenate([idx_res, idx_uni])

        self._active_idx = idx
        self._resample_count += 1

        mean_r = float(self._pool_residuals[idx].mean()) if self._pool_residuals is not None else 0.0
        self._history.append({
            "resample_idx": self._resample_count,
            "n_active": n_active,
            "mean_residual_active": mean_r,
        })
        logger.info(
            f"AdaptiveCollocationSampler: resample #{self._resample_count} — "
            f"n_active={n_active}  mean_residual={mean_r:.3e}"
        )

    def _eval_full_pool(self, model: nn.Module) -> None:
        """Evaluate residual_fn over the full pool in batches."""
        assert self.residual_fn is not None
        b = self.cfg.residual_batch
        all_r = []
        was_training = model.training
        model.eval()
        for start in range(0, self._n_pool, b):
            pts = self._pool[start:start + b]
            with torch.enable_grad():
                r = self.residual_fn(model, pts)
            if r.dim() > 1:
                r = r.pow(2).sum(dim=-1)
            all_r.append(r.detach().cpu().float().numpy())
        if was_training:
            model.train()
        self._pool_residuals = np.concatenate(all_r)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def resample_count(self) -> int:
        return self._resample_count

    def resample_history(self) -> List[Dict]:
        return list(self._history)

    def residual_stats(self) -> Dict[str, float]:
        """Summary statistics of current pool residuals."""
        if self._pool_residuals is None:
            return {}
        r = self._pool_residuals
        return {
            "mean": float(r.mean()),
            "max": float(r.max()),
            "p90": float(np.percentile(r, 90)),
            "p50": float(np.percentile(r, 50)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Channel-scale estimation utilities
# ─────────────────────────────────────────────────────────────────────────────

def estimate_channel_scales(
    Y: torch.Tensor,
    quantile: float = 0.95,
    min_scale: float = 1e-6,
    channel_names: Optional[List[str]] = None,
) -> List[float]:
    """Estimate per-channel output scales as the q-th quantile of |Y_i|.

    This is the "FIX 1" pattern from multi-scale PINN problems: when output
    channels have very different magnitudes (e.g. u_r ~ 0.5 mm vs u_θ ~ 290 mm),
    a shared network struggles to represent all channels equally.  Dividing each
    channel by its typical magnitude during training, then multiplying back on
    output, removes this disparity.

    Parameters
    ----------
    Y               : (N, C) tensor of output values
    quantile        : quantile used to estimate scale (default 0.95)
    min_scale       : floor value to avoid division by zero
    channel_names   : optional names for logging

    Returns
    -------
    List of C float scale factors, one per output channel.
    """
    n_channels = Y.shape[1]
    scales = []
    for i in range(n_channels):
        s = float(torch.quantile(Y[:, i].abs(), quantile).item())
        s = max(s, min_scale)
        scales.append(s)

    if channel_names is not None:
        parts = "  ".join(
            f"{name}={s:.4e}" for name, s in zip(channel_names, scales)
        )
        logger.info(f"Channel scales (p{int(quantile*100)}): {parts}")

    return scales


def filter_nan_rows(
    *tensors: torch.Tensor,
    warn: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Remove rows where any tensor has NaN or Inf.

    Useful for cleaning supervised CFD/FEM datasets before training.

    Parameters
    ----------
    *tensors : one or more (N, ...) tensors that must be filtered together
    warn     : if True, log a warning when rows are removed

    Returns
    -------
    Tuple of filtered tensors (same order as input).
    """
    if not tensors:
        return tensors
    n = tensors[0].shape[0]
    valid = torch.ones(n, dtype=torch.bool, device=tensors[0].device)
    for t in tensors:
        finite = torch.isfinite(t.reshape(n, -1)).all(dim=1)
        valid &= finite

    n_bad = int((~valid).sum().item())
    if n_bad > 0 and warn:
        logger.warning(
            f"filter_nan_rows: removed {n_bad}/{n} rows ({n_bad/n*100:.1f}%) "
            "containing NaN/Inf."
        )
    return tuple(t[valid] for t in tensors)
