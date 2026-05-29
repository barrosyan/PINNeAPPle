"""Shared helpers for preset modules (not part of the public API)."""
from __future__ import annotations


def _lame(E: float, nu: float):
    """Compute Lamé constants (lambda, mu) from Young's modulus and Poisson's ratio."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return float(lam), float(mu)
