"""Second-order TVD (Total Variation Diminishing) upwind scheme for 1D scalar
advection, with a van Leer flux limiter.

Governing equation: df/dt = -v * df/dz  (constant-velocity linear advection;
also usable as the advective term inside a larger operator-split PDE solve).

The van Leer limiter Phi(r) = (r + |r|) / (1 + |r|) blends first-order
upwind (near discontinuities/extrema, where it approaches 0) with a
second-order-accurate correction (in smooth regions, where it approaches 1)
based on the local ratio of successive gradients r — this is what keeps the
scheme free of the spurious oscillations a naive higher-order scheme would
produce near sharp fronts, while still being 2nd-order accurate away from
them. Standard, textbook TVD finite-volume method (e.g. LeVeque, "Finite
Volume Methods for Hyperbolic Problems", Ch. 6 & 9); generic to any 1D
scalar field being advected at a known (possibly time-varying, spatially
uniform) velocity — temperature, concentration, or any other conserved
scalar.
"""
from __future__ import annotations

import numpy as np


def van_leer_limiter(r: np.ndarray) -> np.ndarray:
    """van Leer TVD flux limiter: Phi(r) = (r + |r|) / (1 + |r|). r is the
    ratio of consecutive solution gradients (upstream/downstream); Phi(r)=0
    for r<=0 (a local extremum -- falls back to pure upwind) and Phi(r) -> 1
    as r -> infinity (smooth region -- full 2nd-order correction)."""
    r = np.asarray(r, dtype=np.float64)
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)


def tvd_advection_rhs(f: np.ndarray, v: float, dz: float) -> np.ndarray:
    """Second-order TVD-upwind estimate of df/dt = -v*df/dz on a uniform
    grid of spacing dz, given the field f (shape (N,)) and a constant
    velocity v (v>0: flow toward increasing index; v<0: toward decreasing
    index). Domain-boundary interface fluxes use plain first-order upwind
    (no downstream neighbor available to form a limiter ratio there).

    Returns d f/dt, shape (N,), the advective rate of change at each node
    (== -(flux[i+1] - flux[i])/dz for the reconstructed face fluxes).
    """
    f = np.asarray(f, dtype=np.float64)
    N = f.size
    flux = np.zeros(N + 1)
    eps = 1e-12

    if v >= 0:
        df_up = f[1:] - f[:-1]                       # (N-1,), df_up[i] = f[i]-f[i-1] for i=1..N-1
        df_down = np.empty(N - 1)
        df_down[:-1] = df_up[1:]                      # f[i+1]-f[i] for i=1..N-2
        df_down[-1] = df_up[-1]                        # last interior point: no downstream neighbor
        r = df_down / (df_up + eps * np.sign(df_up + eps))
        phi = van_leer_limiter(r)
        flux[1:N] = v * (f[:-1] + 0.5 * phi * df_up)
        flux[0] = v * f[0]
        flux[N] = v * f[-1]
    else:
        df_up = f[1:] - f[:-1]                        # (N-1,), df_up[i] = f[i+1]-f[i] for i=0..N-2
        df_down = np.empty(N - 1)
        df_down[1:] = df_up[:-1]                       # f[i]-f[i-1] for i=1..N-2
        df_down[0] = df_up[0]
        r = df_down / (df_up + eps * np.sign(df_up + eps))
        phi = van_leer_limiter(r)
        flux[1:N] = v * (f[1:] - 0.5 * phi * df_up)
        flux[0] = v * f[0]
        flux[N] = v * f[-1]

    return -(flux[1:] - flux[:-1]) / dz
