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

``xi`` can be sampled two ways:

- **per-point i.i.d.** (:func:`sample_latent`, the default when no
  ``latent_sampler`` is given to :func:`ensemble_forward`) -- simple, but
  every point's latent code is independent, so individual generated
  realisations are spatially disconnected noise, not coherent structure
  (see below).
- **as a spatially-correlated field** (:class:`CorrelatedLatentField`) --
  draws one or more smooth 3D Gaussian random fields (per-axis anisotropic
  correlation length, periodic- or reflect-padded per axis as appropriate)
  and looks them up at query coordinates, so nearby points share
  correlated ``xi`` and a generated realisation can look like an actual
  coherent flow feature (e.g. a streak) instead of per-point noise. This
  was validated on a real wall-bounded turbulent-channel LES surrogate:
  switching from i.i.d. to a correlated field turned scattered noise into
  streamwise-elongated structure resembling the real flow's streaks (the
  correct anisotropy -- x much longer than y/z -- had to be found
  empirically; an isotropic length scale produced wrongly-oriented,
  wall-normal-elongated streaks instead).

Historical note: this per-point-only limitation used to be a "known,
documented simplification" here with no fix implemented -- :func:`
ensemble_forward`'s optional ``latent_sampler`` parameter and
:class:`CorrelatedLatentField` close that gap while staying fully
backward compatible (``latent_sampler=None`` reproduces the old
per-point-i.i.d. behaviour exactly).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _gaussian_kernel1d(sigma_cells: float, device, dtype) -> Tuple[torch.Tensor, int]:
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / max(sigma_cells, 1e-6)) ** 2)
    k = k / k.sum()
    return k, radius


def _blur_dim(field: torch.Tensor, dim: int, kernel: torch.Tensor, radius: int, periodic: bool) -> torch.Tensor:
    """Separable 1D Gaussian blur of ``field`` along ``dim``. Circular
    padding for a periodic axis, reflect padding otherwise -- convolving
    white noise with this kernel produces a Gaussian random field whose
    boundary behaviour matches the axis (periodic in x/z, wall-bounded in
    y, for a typical channel-flow use case)."""
    n = field.shape[dim]
    r = min(radius, max(n - 1, 0))
    moved = field.movedim(dim, -1)
    shape = moved.shape
    flat = moved.reshape(-1, 1, shape[-1])
    if r > 0:
        pad_mode = "circular" if periodic else "reflect"
        flat = F.pad(flat, (r, r), mode=pad_mode)
        k = kernel if r == radius else _gaussian_kernel1d(max(r / 3.0, 1e-6), field.device, field.dtype)[0]
    else:
        k = kernel
    out = F.conv1d(flat, k.view(1, 1, -1))
    out = out.reshape(*shape[:-1], shape[-1])
    return out.movedim(-1, dim)


class CorrelatedLatentField:
    """Spatially-correlated replacement for per-point :func:`sample_latent`.

    Draws ``n_fields`` independent 3D Gaussian random fields (one per
    latent channel, per field) by convolving white noise with a Gaussian
    kernel -- circular padding along axes named in ``periodic_axes``,
    reflect padding along the rest -- which is exactly the covariance-
    kernel-via-convolution construction of a GRF. :meth:`lookup` then
    trilinearly interpolates a drawn field at arbitrary physical query
    coordinates, so every point in a batch queried against the *same*
    field draw gets correlated ``xi`` -- nearby points get similar values,
    so the network can learn to associate a smoothly-varying ``xi`` region
    with a coherent output structure, instead of only ever seeing
    per-point noise (see :func:`sample_latent`/module docstring).

    ``length_scale`` sets the correlation length per axis (same physical
    units as ``bounds``) -- either one scalar (isotropic) or a
    ``{axis: value}`` dict / sequence matching ``spatial_axes`` order.
    Real turbulent flow features are often strongly anisotropic (e.g. a
    channel flow's streamwise streak length is roughly an order of
    magnitude longer than its spanwise spacing or wall-normal extent) --
    an isotropic scale gives structure the wrong shape/orientation, not
    just the wrong size, so prefer per-axis values whenever known.

    Parameters
    ----------
    bounds : ``{axis: (lo, hi)}`` for every axis in ``spatial_axes``.
    coord_order : full column order of query points passed to
        :meth:`lookup` (may include a time axis; only entries also present
        in ``bounds``/``spatial_axes`` are used for the lookup).
    spatial_axes : the (exactly 3) spatial axis names this field spans, in
        the order its grid dimensions are laid out.
    periodic_axes : subset of ``spatial_axes`` treated as periodic.
    latent_dim : number of independent latent channels per field.
    grid_res : coarse grid resolution per spatial axis.
    """

    def __init__(self, bounds: Dict[str, Sequence[float]], coord_order: Sequence[str],
                 spatial_axes: Sequence[str], periodic_axes: Sequence[str], latent_dim: int,
                 grid_res: Sequence[int] = (64, 24, 32), length_scale=0.75):
        self.coord_order = list(coord_order)
        self.spatial_axes = list(spatial_axes)
        if len(self.spatial_axes) != 3 or len(grid_res) != 3:
            raise ValueError("CorrelatedLatentField assumes exactly 3 spatial axes")
        self.bounds = {a: (float(bounds[a][0]), float(bounds[a][1])) for a in self.spatial_axes}
        self.periodic = [a in set(periodic_axes) for a in self.spatial_axes]
        self.latent_dim = int(latent_dim)
        self.grid_res = tuple(int(r) for r in grid_res)
        if isinstance(length_scale, dict):
            self.length_scale = {a: float(length_scale[a]) for a in self.spatial_axes}
        elif isinstance(length_scale, (int, float)):
            self.length_scale = {a: float(length_scale) for a in self.spatial_axes}
        else:
            self.length_scale = {a: float(v) for a, v in zip(self.spatial_axes, list(length_scale))}

    def sample_field(self, n_fields: int, device, dtype=torch.float32) -> torch.Tensor:
        """Draw ``n_fields`` independent correlated fields: (n_fields, latent_dim, *grid_res)."""
        field = torch.randn(n_fields, self.latent_dim, *self.grid_res, device=device, dtype=dtype)
        for i, ax in enumerate(self.spatial_axes):
            lo, hi = self.bounds[ax]
            dx = (hi - lo) / self.grid_res[i]
            sigma_cells = max(self.length_scale[ax] / dx, 1e-6)
            kernel, radius = _gaussian_kernel1d(sigma_cells, device, dtype)
            field = _blur_dim(field, 2 + i, kernel, radius, self.periodic[i])
        dims = (2, 3, 4)
        mean = field.mean(dim=dims, keepdim=True)
        std = field.std(dim=dims, keepdim=True).clamp_min(1e-6)
        return (field - mean) / std

    def lookup(self, field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of ``field`` (n_fields, latent_dim, *grid_res)
        at physical query points ``x`` (N, len(coord_order)). Returns
        (N, n_fields, latent_dim)."""
        idx_map = {name: i for i, name in enumerate(self.coord_order)}
        K, L = field.shape[0], field.shape[1]
        N = x.shape[0]
        i0, i1, frac = [], [], []
        for i, ax in enumerate(self.spatial_axes):
            lo, hi = self.bounds[ax]
            n = self.grid_res[i]
            col = x[:, idx_map[ax]]
            u = (col - lo) / (hi - lo)
            if self.periodic[i]:
                u = u - torch.floor(u)
                fi = u * n
                lo_idx = torch.floor(fi).long() % n
                hi_idx = (lo_idx + 1) % n
            else:
                fi = u.clamp(0.0, 1.0) * (n - 1)
                lo_idx = torch.floor(fi).long().clamp(0, n - 1)
                hi_idx = (lo_idx + 1).clamp(max=n - 1)
            i0.append(lo_idx)
            i1.append(hi_idx)
            frac.append(fi - torch.floor(fi))
        out = x.new_zeros(N, K, L)
        for c0 in (0, 1):
            for c1 in (0, 1):
                for c2 in (0, 1):
                    ia = i1[0] if c0 else i0[0]
                    ib = i1[1] if c1 else i0[1]
                    ic = i1[2] if c2 else i0[2]
                    w = ((frac[0] if c0 else (1.0 - frac[0])) *
                         (frac[1] if c1 else (1.0 - frac[1])) *
                         (frac[2] if c2 else (1.0 - frac[2])))
                    vals = field[:, :, ia, ib, ic]  # (K, L, N)
                    out = out + w.view(N, 1, 1) * vals.permute(2, 0, 1)
        return out

    def sample_xi(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw fresh field(s) and return ``xi`` at every point in ``x``:
        (N, n_samples, latent_dim). One call = one set of coherent
        realisation(s) shared across all points in ``x``."""
        field = self.sample_field(n_samples, x.device, x.dtype)
        return self.lookup(field, x)


def ensemble_forward(model, x: torch.Tensor, latent_dim: int, n_samples: int,
                      latent_sampler: Optional["CorrelatedLatentField"] = None) -> torch.Tensor:
    """Evaluate ``n_samples`` latent draws at every point in ``x`` (N, D) in
    a single batched forward pass. Returns ``(N, n_samples, out_dim)``.

    ``latent_sampler=None`` (default): each point gets its own independent
    set of draws -- see the module docstring for why that is a real
    limitation for generating individually-coherent realisations (the
    ensemble mean/covariance losses are unaffected either way, since they
    are simple Monte Carlo point statistics regardless of spatial
    correlation).

    ``latent_sampler`` given (a :class:`CorrelatedLatentField`): the
    ``n_samples`` draws are ``n_samples`` independent *coherent fields*,
    each queried at every point in ``x`` -- nearby points share correlated
    ``xi`` within one realisation.
    """
    n = x.shape[0]
    x_tiled = x.repeat_interleave(n_samples, dim=0)
    if latent_sampler is not None:
        xi = latent_sampler.sample_xi(x, n_samples=n_samples).reshape(n * n_samples, latent_dim)
    else:
        xi = sample_latent(n * n_samples, latent_dim, x.device, x.dtype)
    y = model(x_tiled, xi)
    return y.reshape(n, n_samples, y.shape[-1])


#: Covariance-pair preset matching OpenFOAM's own ``symmTensor`` component
#: order (xx, xy, xz, yy, yz, zz) -- the convention ``UPrime2Mean`` (and any
#: other OpenFOAM Reynolds-stress-like field) is written in, so a CFD
#: surrogate reading such a field straight off disk can pass this directly
#: as ``cov_index_pairs`` to :func:`mean_covariance_loss` without having to
#: work out the index pairing by hand.
OPENFOAM_SYMM_TENSOR_PAIRS: Sequence[Tuple[int, int]] = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def mean_covariance_loss(
    model,
    x: torch.Tensor,
    latent_dim: int,
    n_samples: int,
    mean_target: torch.Tensor,
    cov_target: Optional[torch.Tensor] = None,
    cov_index_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    field_slice: slice = slice(None),
    latent_sampler: Optional["CorrelatedLatentField"] = None,
    ens: Optional[torch.Tensor] = None,
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
    latent_sampler : optional :class:`CorrelatedLatentField` -- forwarded
        to :func:`ensemble_forward` unchanged; ``None`` (default) keeps the
        original per-point-i.i.d. ensemble. Ignored if ``ens`` is given.
    ens : optional pre-computed ``(N, n_samples, out_dim)`` ensemble (from
        a prior :func:`ensemble_forward` call) -- skips this function's own
        internal ``ensemble_forward`` call when given. Useful when a
        caller needs several different statistics (e.g. a velocity mean +
        covariance here, plus some other field's mean matched separately)
        from the *same* ensemble draw: calling this twice without ``ens``
        would draw two independent ensembles, which is both wasteful and
        statistically inconsistent between the two matched quantities.

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
    if ens is None:
        ens = ensemble_forward(model, x, latent_dim, n_samples, latent_sampler=latent_sampler)  # (N, K, out_dim)
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
    "CorrelatedLatentField",
    "ensemble_forward",
    "mean_covariance_loss",
    "OPENFOAM_SYMM_TENSOR_PAIRS",
]
