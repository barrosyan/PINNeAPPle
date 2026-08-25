"""
Design-of-experiments parameter-space sampling.

Generic helpers for sampling a bounded parameter space — e.g. to build a
synthetic dataset by sweeping a physical problem's parameters, or to run a
global sensitivity analysis. Distinct from `collocation.py`/`active_learning.py`
in this package, which sample *collocation points in the PDE domain*; this
module samples *parameter values* (physical constants, boundary condition
magnitudes, geometry dimensions, etc.).

Built on `scipy.stats.qmc` — no extra dependency beyond scipy.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import qmc

SamplingMethod = Literal["lhs", "sobol", "random", "grid"]


def sample_parameters(
    param_ranges: dict[str, tuple[float, float]],
    n_samples: int,
    method: SamplingMethod = "lhs",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Sample `n_samples` points from a bounded parameter space.

    Parameters
    ----------
    param_ranges : {name: (low, high)} for each parameter to sample.
    n_samples : number of samples to draw.
    method : "lhs" (Latin Hypercube, default — good space-filling for a
        moderate sample budget), "sobol" (low-discrepancy sequence, best for
        large sample counts / convergence studies), "random" (plain uniform),
        or "grid" (regular grid, exponential in dimension — use only for
        low-dimensional sweeps).
    seed : RNG seed for reproducibility.

    Returns
    -------
    A list of `n_samples` dicts, one key per `param_ranges` entry.
    """
    names = list(param_ranges.keys())
    d = len(names)
    if d == 0:
        return [{} for _ in range(n_samples)]

    lows = np.array([param_ranges[n][0] for n in names])
    highs = np.array([param_ranges[n][1] for n in names])

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        unit = sampler.random(n=n_samples)
    elif method == "sobol":
        sampler = qmc.Sobol(d=d, seed=seed)
        m = int(np.ceil(np.log2(max(n_samples, 1))))
        unit = sampler.random_base2(m=m)[:n_samples]
    elif method == "random":
        rng = np.random.default_rng(seed)
        unit = rng.random((n_samples, d))
    elif method == "grid":
        per_dim = max(int(round(n_samples ** (1.0 / d))), 1)
        axes = [np.linspace(0.0, 1.0, per_dim) for _ in range(d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        unit = np.stack([m.ravel() for m in mesh], axis=-1)[:n_samples]
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    scaled = qmc.scale(unit, lows, highs)
    return [dict(zip(names, row)) for row in scaled]


def saltelli_perturbation_sweep(
    nominal: dict[str, float],
    param_ranges: dict[str, tuple[float, float]],
    n_base: int = 8,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Build a Saltelli-style sample set for first-order/total-order Sobol
    global sensitivity analysis, without requiring the SALib dependency.

    Produces 2*n_base_rounded*(d+2)-ish samples: a base matrix A, a
    resampling matrix B, and d "swap one column of A with B" matrices —
    the standard construction needed to estimate first-order and total-order
    Sobol indices from model evaluations on this sample set.

    `nominal` is accepted for interface symmetry with call sites that also
    need a baseline point, but is not itself perturbed here — sensitivity
    indices are computed downstream from model outputs on the returned
    sample set, not from this function.
    """
    names = list(param_ranges.keys())
    d = len(names)
    lows = np.array([param_ranges[n][0] for n in names])
    highs = np.array([param_ranges[n][1] for n in names])

    m = int(np.ceil(np.log2(max(n_base, 1))))
    sampler = qmc.Sobol(d=2 * d, seed=seed)
    unit = sampler.random_base2(m=m)
    a, b = unit[:, :d], unit[:, d:]

    samples = []
    for row in np.vstack([a, b]):
        scaled = qmc.scale(row.reshape(1, -1), lows, highs)[0]
        samples.append(dict(zip(names, scaled)))

    for i in range(d):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        for row in ab_i:
            scaled = qmc.scale(row.reshape(1, -1), lows, highs)[0]
            samples.append(dict(zip(names, scaled)))

    return samples
