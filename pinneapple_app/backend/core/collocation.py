"""Collocation point generation strategies for pinneapple_app."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


STRATEGIES = ["lhs", "sobol", "uniform", "grid", "halton", "adaptive", "meshfree"]

STRATEGY_LABELS = {
    "lhs":      "Latin Hypercube Sampling",
    "sobol":    "Sobol quasi-random",
    "uniform":  "Uniform random",
    "grid":     "Structured grid",
    "halton":   "Halton sequence",
    "adaptive": "Residual-based adaptive",
    "meshfree": "Meshfree (RBF)",
}


@dataclass
class CollocationConfig:
    """Configuration for collocation point generation."""
    strategy: str = "lhs"               # one of STRATEGIES
    n_interior: int = 8192
    n_boundary: int = 1024
    n_initial: int = 512                # ICs (time-dependent problems)
    seed: int = 42
    # geometry-aware sampling (optional)
    use_geometry: bool = False
    geometry_sdf_fn: Optional[object] = None  # callable(pts) -> signed distances
    domain_name: Optional[str] = None
    # grid settings
    grid_resolution: int = 64           # per dimension (for "grid" strategy)
    # adaptive refinement
    n_adaptive_rounds: int = 3
    adaptive_threshold: float = 0.1


def generate_points(
    config: CollocationConfig,
    bounds: Dict[str, Tuple[float, float]],
    *,
    include_time: bool = False,
    t_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, np.ndarray]:
    """Generate collocation points from a config and domain bounds.

    Returns
    -------
    dict with keys:
        "interior"  : (N_int, D) — interior collocation points
        "boundary"  : (N_bnd, D+1?) — boundary points + BC label column
        "initial"   : (N_ic, D) — initial-condition points (if time-dependent)
    """
    coord_names = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in coord_names], dtype=np.float32)
    hi = np.array([bounds[k][1] for k in coord_names], dtype=np.float32)
    dim = len(coord_names)

    rng = np.random.default_rng(config.seed)

    if config.use_geometry and config.geometry_sdf_fn is not None:
        x_int = _sample_via_sdf(
            config.geometry_sdf_fn, lo, hi, config.n_interior,
            rng, config.strategy
        )
    else:
        x_int = _sample_interior(
            config.strategy, lo, hi, config.n_interior, rng,
            config.grid_resolution, dim
        )

    x_bnd = _sample_boundary(bounds, config.n_boundary, rng, include_time, t_range)

    result = {"interior": x_int, "boundary": x_bnd}

    if include_time:
        x_ic = _sample_interior(
            "lhs", lo[:dim], hi[:dim], config.n_initial, rng, config.grid_resolution, dim
        )
        # set time column = t_range[0]
        t_col = np.full((len(x_ic), 1), t_range[0], dtype=np.float32)
        result["initial"] = np.concatenate([x_ic, t_col], axis=1)

    return result


# ── Sampling helpers ──────────────────────────────────────────────────────

def _sample_interior(
    strategy: str,
    lo: np.ndarray,
    hi: np.ndarray,
    n: int,
    rng: np.random.Generator,
    grid_res: int,
    dim: int,
) -> np.ndarray:
    if strategy == "uniform":
        pts = rng.uniform(0, 1, (n, dim)).astype(np.float32)
    elif strategy == "lhs":
        pts = _lhs(n, dim, rng)
    elif strategy == "sobol":
        pts = _sobol(n, dim)
    elif strategy == "halton":
        pts = _halton(n, dim)
    elif strategy == "grid":
        pts = _grid(dim, grid_res)
        if len(pts) > n:
            idx = rng.choice(len(pts), n, replace=False)
            pts = pts[idx]
    else:
        pts = _lhs(n, dim, rng)

    return (lo + pts * (hi - lo)).astype(np.float32)


def _sample_boundary(
    bounds: Dict[str, Tuple[float, float]],
    n: int,
    rng: np.random.Generator,
    include_time: bool,
    t_range: Tuple[float, float],
) -> np.ndarray:
    """Sample points on each face of the hyperbox domain."""
    coord_names = list(bounds.keys())
    spatial_names = [c for c in coord_names if c != "t"]
    dim = len(spatial_names)
    n_faces = 2 * dim
    n_per_face = max(1, n // n_faces)

    lo = np.array([bounds[k][0] for k in spatial_names], dtype=np.float32)
    hi = np.array([bounds[k][1] for k in spatial_names], dtype=np.float32)

    pts_list = []
    for d in range(dim):
        for val in (lo[d], hi[d]):
            base = rng.uniform(0, 1, (n_per_face, dim)).astype(np.float32)
            base = lo + base * (hi - lo)
            base[:, d] = val
            pts_list.append(base)

    pts = np.concatenate(pts_list, axis=0)

    if include_time:
        t_col = rng.uniform(t_range[0], t_range[1], (len(pts), 1)).astype(np.float32)
        pts = np.concatenate([pts, t_col], axis=1)

    return pts


def _sample_via_sdf(
    sdf_fn,
    lo: np.ndarray,
    hi: np.ndarray,
    n: int,
    rng: np.random.Generator,
    strategy: str,
) -> np.ndarray:
    """Rejection-sample points inside an SDF-defined geometry."""
    dim = len(lo)
    collected = []
    batch = n * 10
    while sum(len(c) for c in collected) < n:
        candidates = lo + rng.uniform(0, 1, (batch, dim)).astype(np.float32) * (hi - lo)
        sdf_vals = np.asarray(sdf_fn(candidates))
        inside = candidates[sdf_vals <= 0]
        collected.append(inside)
    pts = np.concatenate(collected, axis=0)[:n]
    return pts


def _lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    pts = np.zeros((n, dim), dtype=np.float32)
    for d in range(dim):
        perm = rng.permutation(n)
        pts[:, d] = (perm + rng.uniform(0, 1, n)) / n
    return pts


def _sobol(n: int, dim: int) -> np.ndarray:
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=dim, scramble=True)
        m = int(np.ceil(np.log2(n)))
        pts = sampler.random_base2(m=m)[:n]
        return pts.astype(np.float32)
    except ImportError:
        rng = np.random.default_rng(0)
        return _lhs(n, dim, rng)


def _halton(n: int, dim: int) -> np.ndarray:
    try:
        from scipy.stats.qmc import Halton
        sampler = Halton(d=dim, scramble=True)
        return sampler.random(n).astype(np.float32)
    except ImportError:
        rng = np.random.default_rng(0)
        return _lhs(n, dim, rng)


def _grid(dim: int, res: int) -> np.ndarray:
    axes = [np.linspace(0, 1, res, dtype=np.float32) for _ in range(dim)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)
