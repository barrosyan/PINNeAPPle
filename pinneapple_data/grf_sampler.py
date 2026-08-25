"""Gaussian Random Field (GRF) input-function sampler, the standard way to
draw random forcing/coefficient functions u(x) for operator-learning
benchmarks (e.g. Lu, Jin & Karniadakis 2019, "DeepONet", arXiv:1910.03193,
which draws every one of its input functions from a zero-mean GRF with an
RBF/squared-exponential covariance kernel, length scale 0.2 by default).
Useful any time a dataset needs many independent random functions sharing a
common smoothness/correlation structure — not just for DeepONet.

Design (1D, `sample_grf_batch`): one Cholesky factorization of the RBF
covariance on a fixed fine grid (n_grid points) per BATCH of draws, not one
per draw — draws are just L @ standard_normal, and each draw is wrapped in a
CubicSpline so downstream ODE/PDE solvers can query u(x) at whatever points
their own integrator lands on, not just the n_grid grid nodes (this is the
literature-standard implementation approach — DeepXDE's own GRF class does
the same thing: a spline over a fine grid, not GP-reconditioning at
arbitrary query time, which would be O(m^3) per query batch instead of
O(n_grid) once per draw).

Design (2D, `sample_grf_2d_batch`): a direct 2D generalization of the 1D
approach (Cholesky of the full covariance matrix) would need an (nx*ny) x
(nx*ny) covariance matrix — O((nx*ny)^3) Cholesky, e.g. a modest 64x64 grid
is already a 4096x4096 factorization per draw. Instead this uses the
standard spectral-synthesis / circulant-embedding-adjacent FFT method for
stationary GRFs (Wiener-Khinchin theorem: the power spectral density of a
stationary random field is the Fourier transform of its covariance
function): draw white noise on an (nx, ny) grid, multiply its 2D FFT by the
square root of the target covariance kernel's power spectral density,
inverse-FFT, take the real part. This is O(nx*ny*log(nx*ny)) per draw and is
a named, textbook method (see e.g. Lang & Potthoff 2011, "Fast simulation of
Gaussian random fields"), not hand-rolled.

The covariance kernel matched is the same RBF/squared-exponential family
used by the 1D sampler: C(r) = exp(-r^2 / (2*l^2)). Its 2D continuous
Fourier transform is itself Gaussian in wavenumber space, so sqrt(PSD(k)) ~
exp(-l^2*|k|^2/4) up to an overall multiplicative constant — that constant
is irrelevant here because every draw is renormalized to zero mean / unit
variance post-synthesis (the discrete FFT's own normalization convention
and grid resolution both affect the raw variance, so matching a target
variance by construction is fragile; renormalizing empirically is the
standard, robust fix, since only the spatial correlation structure matters
for most downstream uses, not an absolute scale — callers apply their own
amplitude/sigma multiplier on top).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator


@dataclass
class GRFDraw:
    """One sampled function u(x), x in [x_min, x_max]."""
    x_grid: np.ndarray      # (n_grid,) -- the fine grid the spline was built from
    u_grid: np.ndarray      # (n_grid,) -- u evaluated at x_grid
    spline: CubicSpline     # callable: spline(x) -> u(x) for arbitrary x, and
                             # spline(x, 1) / spline(x, 2) for derivatives

    def __call__(self, x: "Union[np.ndarray, float]") -> np.ndarray:
        return self.spline(x)

    def sensors(self, x_sensors: np.ndarray) -> np.ndarray:
        """u evaluated at a fixed set of sensor points (an operator-learning
        branch net's input — m points, conventionally evenly spaced)."""
        return self.spline(x_sensors)


def _rbf_covariance(x_grid: np.ndarray, length_scale: float) -> np.ndarray:
    diff = x_grid[:, None] - x_grid[None, :]
    return np.exp(-0.5 * (diff / length_scale) ** 2)


def sample_grf_batch(
    n_draws: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_grid: int = 1000,
    length_scale: float = 0.2,
    jitter: float = 1e-10,
    seed: Optional[int] = None,
) -> List[GRFDraw]:
    """Draw n_draws independent zero-mean GRF paths sharing ONE Cholesky
    factorization of the RBF covariance on a common fine grid.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(x_min, x_max, n_grid)
    cov = _rbf_covariance(x_grid, length_scale)
    cov[np.diag_indices(n_grid)] += jitter
    L = np.linalg.cholesky(cov)

    z = rng.standard_normal((n_grid, n_draws))
    u_batch = L @ z  # (n_grid, n_draws)

    draws = []
    for i in range(n_draws):
        u_grid = u_batch[:, i]
        spline = CubicSpline(x_grid, u_grid)
        draws.append(GRFDraw(x_grid=x_grid, u_grid=u_grid, spline=spline))
    return draws


def sample_grf_paths(
    n_draws: int,
    x_min: float = 0.0,
    x_max: float = 1.0,
    n_grid: int = 1000,
    length_scale: float = 0.2,
    jitter: float = 1e-10,
    seed: Optional[int] = None,
) -> List[GRFDraw]:
    """Alias for sample_grf_batch (kept for readability at call sites that
    draw a single small batch rather than a full dataset-generation pass)."""
    return sample_grf_batch(n_draws, x_min, x_max, n_grid, length_scale, jitter, seed)


@dataclass
class GRFDraw2D:
    """One sampled 2D field u(x,y), (x,y) in [0,1]x[0,1] (normalized)."""
    x_grid: np.ndarray       # (nx,) -- the FFT synthesis grid's x-coordinates
    y_grid: np.ndarray       # (ny,) -- the FFT synthesis grid's y-coordinates
    u_grid: np.ndarray       # (nx, ny) -- u evaluated on the synthesis grid
    interp: RegularGridInterpolator  # callable at arbitrary (x,y) points

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        """xy: (N, 2) array of (x,y) points -> (N,) array of u(x,y)."""
        return self.interp(np.asarray(xy))

    def sensors(self, x_sensors: np.ndarray, y_sensors: np.ndarray) -> np.ndarray:
        """u evaluated on a fixed m_x-by-m_y sensor grid (an operator-learning
        branch net's input), flattened row-major (x-major, matching
        indexing='ij'): length m_x*m_y."""
        Xs, Ys = np.meshgrid(x_sensors, y_sensors, indexing="ij")
        pts = np.stack([Xs.ravel(), Ys.ravel()], axis=1)
        return self.interp(pts)


def _rbf_psd_sqrt_2d(kx: np.ndarray, ky: np.ndarray, length_scale: float) -> np.ndarray:
    k2 = kx ** 2 + ky ** 2
    return np.exp(-0.25 * (length_scale ** 2) * k2)


def sample_grf_2d_batch(
    n_draws: int,
    nx: int = 64,
    ny: int = 64,
    length_scale: float = 0.2,
    seed: Optional[int] = None,
) -> List[GRFDraw2D]:
    """Draw n_draws independent zero-mean, unit-variance 2D GRF paths on the
    normalized [0,1]x[0,1] domain via FFT spectral synthesis (see module
    docstring above). One (nx,ny) white-noise draw + one 2D FFT + one
    inverse 2D FFT per sample — no shared factorization to amortize across
    draws the way sample_grf_batch's Cholesky is (the FFT method has no
    equivalent expensive-precompute step to share), so batching here is just
    a loop, not a performance-motivated primitive like the 1D version.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(0.0, 1.0, nx)
    y_grid = np.linspace(0.0, 1.0, ny)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    sqrt_psd = _rbf_psd_sqrt_2d(KX, KY, length_scale)

    draws = []
    for _ in range(n_draws):
        noise = rng.standard_normal((nx, ny))
        field_hat = np.fft.fft2(noise) * sqrt_psd
        field = np.real(np.fft.ifft2(field_hat))
        field = (field - field.mean()) / (field.std() + 1e-12)
        interp = RegularGridInterpolator((x_grid, y_grid), field, method="linear",
                                          bounds_error=False, fill_value=None)
        draws.append(GRFDraw2D(x_grid=x_grid, y_grid=y_grid, u_grid=field, interp=interp))
    return draws
