"""Latent-conditioned (stochastic) PINN training utilities.

A deterministic PINN, ``f(x) -> y``, is the wrong tool for a chaotic
system with only a handful of real observed snapshots to fit against: with
too few, decorrelated snapshots to learn genuine dynamics from, a
pointwise-MSE-trained deterministic network is pushed towards either an
over-smoothed, near-constant predictor (nothing left to distinguish
different points) or an overfit one that memorises the specific noisy
values of whatever snapshots it saw, at the cost of any coherent spatial
structure (observed in practice on a wall-bounded turbulent-channel LES
surrogate: a deterministic model trained on two real, 36-time-units-apart
instantaneous snapshots produced non-physical, spatially disconnected
low-velocity blobs scattered through the flow's core, rather than the
wall-hugging streaky structure the real LES has).

This module instead makes the model take an additional latent code ``xi``
(typically ``xi ~ N(0, I)``), turning ``f(x) -> y`` into a family
``f(x, xi) -> y`` -- a *distribution* of physically-plausible outputs at
each ``x``, trained so that:

- every ``(x, xi)`` still satisfies the governing PDE residual and any
  boundary conditions (physics must hold for *every* plausible
  realisation, not just an average one) -- pass ``xi`` straight through to
  whatever residual function you already use, since the residual only
  needs `torch.autograd.grad` w.r.t. the coordinates ``x``, never ``xi``;
- one or more *known* real observations can be anchored to a fixed
  reference code (e.g. ``xi = 0``), so the model reproduces at least the
  literal trajectories you actually have data for;
- the *ensemble* mean and covariance over many random ``xi`` draws
  (:func:`ensemble_forward`, :func:`mean_covariance_loss`) are matched
  against real time/plane-averaged statistics (a mean field and, e.g., a
  Reynolds-stress-like covariance tensor for CFD) -- i.e. the model is
  trained to match the real system's *statistics*, not to hit one noisy
  pointwise value.

Known, documented simplification: ``xi`` here is sampled *independently
per point* (:func:`ensemble_forward` draws a fresh ``xi`` for every row of
the input batch), not as a single spatially-correlated random field shared
across a whole coherent region. A proper turbulence/field generator would
reuse the *same* ``xi`` (or a spatially-correlated latent field) across an
entire coherent structure, so that a generated flow feature is a spatially
extended object rather than a per-point coin flip. That is a real
architecture change (e.g. a convolutional/graph latent field, or
conditioning on a spatially-correlated Gaussian-process noise map) beyond
what this module provides; the mean/covariance-matching losses below are
still meaningful with per-point ``xi`` (they are simple Monte Carlo point
statistics either way), but do not expect an individual generated
realisation to look like a coherent structure the way real instantaneous
snapshots do -- only the *ensemble statistics* are trained to match.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class LatentConditionedModel(nn.Module):
    """Wrap any ``model(x) -> y`` so it additionally accepts a latent code
    ``xi``, concatenated onto ``x`` before the wrapped model's own forward.

    ``latent_dim=0`` degenerates to calling the base model unchanged
    (``xi`` is accepted but ignored), so wrapping is a no-op for anyone not
    using the stochastic mechanism. The wrapped model's ``in_dim`` must be
    ``x.shape[1] + latent_dim``.

    Examples
    --------
    >>> base = ModelRegistry.build("modified_mlp", in_dim=4+8, out_dim=4, ...)
    >>> model = LatentConditionedModel(base, latent_dim=8)
    >>> xi = sample_latent(x.shape[0], latent_dim=8, device=x.device)
    >>> y = model(x, xi)
    """

    def __init__(self, base: nn.Module, latent_dim: int) -> None:
        super().__init__()
        self.base = base
        self.latent_dim = int(latent_dim)

    def forward(self, x: torch.Tensor, xi: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.latent_dim <= 0:
            return self.base(x)
        if xi is None:
            xi = torch.zeros(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        return self.base(torch.cat([x, xi], dim=1))


def sample_latent(n: int, latent_dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``N(0, I)`` latent codes, shape ``(n, latent_dim)``."""
    return torch.randn(n, latent_dim, device=device, dtype=dtype)


def ensemble_forward(model, x: torch.Tensor, latent_dim: int, n_samples: int) -> torch.Tensor:
    """Evaluate ``n_samples`` independent latent draws at every point in
    ``x`` (N, D) in a single batched forward pass.

    Returns ``(N, n_samples, out_dim)``. Each point gets its own
    independent set of draws -- see the module docstring for why that is
    an intentional, documented simplification rather than a
    spatially-correlated random field.
    """
    n = x.shape[0]
    x_tiled = x.repeat_interleave(n_samples, dim=0)
    xi = sample_latent(n * n_samples, latent_dim, x.device, x.dtype)
    y = model(x_tiled, xi)
    return y.reshape(n, n_samples, y.shape[-1])


def mean_covariance_loss(
    model,
    x: torch.Tensor,
    latent_dim: int,
    n_samples: int,
    mean_target: torch.Tensor,
    cov_target: Optional[torch.Tensor] = None,
    cov_index_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    field_slice: slice = slice(None),
) -> Dict[str, torch.Tensor]:
    """Monte-Carlo mean and (optionally) covariance of the model's own
    latent ensemble at ``x``, matched (MSE) against externally supplied
    real statistics.

    Parameters
    ----------
    mean_target : (N, k) real time/plane-averaged field, ``k`` = the
        number of output components selected by ``field_slice``.
    cov_target : (N, len(cov_index_pairs)) optional real covariance-like
        target -- e.g. a Reynolds-stress tensor's independent components.
    cov_index_pairs : which (component_i, component_j) pairs of the
        selected fields to compute covariance for, in the same order as
        ``cov_target``'s columns. Defaults to every unique pair
        (upper-triangular, including the diagonal / variances) if
        ``cov_target`` is given but this is not.
    field_slice : which output columns of the model count as "the field"
        being matched (e.g. ``slice(0, 3)`` for a 3-component velocity out
        of a ``(u, v, w, p)`` output) -- everything else is still computed
        by the ensemble forward pass but ignored by this loss.

    Returns
    -------
    dict with key ``"mean"`` (always) and ``"covariance"`` (only if
    ``cov_target`` is given).

    Examples
    --------
    Matching a CFD mean velocity + Reynolds-stress tensor (OpenFOAM's own
    ``symmTensor`` component order: xx, xy, xz, yy, yz, zz)::

        stats = mean_covariance_loss(
            model, x, latent_dim=8, n_samples=8,
            mean_target=u_mean,                        # (N, 3)
            cov_target=reynolds_stress,                 # (N, 6)
            cov_index_pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)],
            field_slice=slice(0, 3),
        )
        loss = stats["mean"] + stats["covariance"]
    """
    ens = ensemble_forward(model, x, latent_dim, n_samples)  # (N, K, out_dim)
    sel = ens[:, :, field_slice]
    mean_pred = sel.mean(dim=1)
    out = {"mean": torch.mean((mean_pred - mean_target) ** 2)}

    if cov_target is not None:
        k = sel.shape[-1]
        pairs = cov_index_pairs or [(i, j) for i in range(k) for j in range(i, k)]
        fluct = sel - mean_pred.unsqueeze(1)  # (N, K, k)
        denom = max(sel.shape[1] - 1, 1)
        cols = [(fluct[:, :, i] * fluct[:, :, j]).sum(dim=1) / denom for (i, j) in pairs]
        cov_pred = torch.stack(cols, dim=1)  # (N, len(pairs))
        out["covariance"] = torch.mean((cov_pred - cov_target) ** 2)

    return out


__all__ = [
    "LatentConditionedModel",
    "sample_latent",
    "ensemble_forward",
    "mean_covariance_loss",
]
