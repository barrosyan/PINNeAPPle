"""Automated collocation point generation for PINN training.

Provides flexible, strategy-aware sampling from:
- Rectangular/hyperrectangular domains (from bounds or ProblemSpec)
- SDF-defined domains (from pinneaple_geom)
- Mesh-defined domains (from Mesh2D / MeshData)
- Pre-built PhysicsDomain2D objects

Sampling strategies:
- uniform   : independent uniform random
- lhs       : Latin Hypercube Sampling (better space-filling)
- sobol     : Sobol quasi-random sequence (very uniform)
- adaptive  : residual-based adaptive sampling (requires a trained model)

All outputs are numpy float32 arrays in a standard dict-batch format
compatible with the PINN training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _sample_uniform(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random in hypercube. bounds: (D, 2)."""
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (lo + rng.random((n, d)) * (hi - lo)).astype(np.float32)


def _sample_lhs(bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling. Better coverage than uniform."""
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    result = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (lo[j] + (perm + rng.random(n)) / n * (hi[j] - lo[j])).astype(np.float32)
    return result


def _sample_sobol(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Sobol quasi-random sequence. Most uniform for moderate n."""
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=d, scramble=True, seed=seed)
        # Sobol requires n = 2^k
        import math
        k = math.ceil(math.log2(max(n, 2)))
        pts = sampler.random_base2(k)[:n]
        return (lo + pts * (hi - lo)).astype(np.float32)
    except ImportError:
        # Fall back to LHS if scipy not available
        rng = np.random.default_rng(seed)
        return _sample_lhs(bounds, n, rng)


def _sample_adaptive(
    bounds: np.ndarray,
    n: int,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_candidates: int = None,
) -> np.ndarray:
    """Adaptive sampling: sample more densely where PDE residual is high.

    Parameters
    ----------
    residual_fn : callable (N, D) -> (N,) returning |residual| per point
    n_candidates : candidate pool size (default: 10*n)
    """
    if n_candidates is None:
        n_candidates = 10 * n
    candidates = _sample_uniform(bounds, n_candidates, rng)
    res = np.abs(residual_fn(candidates))
    res = res - res.min() + 1e-10
    probs = res / res.sum()
    idx = rng.choice(n_candidates, size=n, replace=(n > n_candidates), p=probs)
    return candidates[idx]


# ---------------------------------------------------------------------------
# Main CollocationSampler class
# ---------------------------------------------------------------------------

@dataclass
class CollocationConfig:
    """Configuration for collocation point generation."""
    n_col: int = 10_000
    n_bc: int = 2_000
    n_ic: int = 1_000
    n_data: int = 0
    strategy: str = "lhs"     # uniform | lhs | sobol | adaptive
    seed: int = 0
    oversample_bc: int = 5    # multiplier for rejection-based boundary sampling
    include_time: bool = False
    t_range: Tuple[float, float] = (0.0, 1.0)


class CollocationSampler:
    """Unified collocation point generator for PINN training.

    Supports multiple domain sources and sampling strategies.
    All outputs are in the standard pinneaple batch dict format.

    Usage examples::

        # From ProblemSpec
        from pinneaple_environment import get_preset
        spec = get_preset("burgers_1d", nu=0.01)
        sampler = CollocationSampler.from_problem_spec(spec)
        batch = sampler.sample(n_col=4096, n_bc=512)

        # From PhysicsDomain2D
        from pinneaple_geom.gen.domains import ChannelWithObstacleDomain2D
        domain = ChannelWithObstacleDomain2D(length=2.0)
        sampler = CollocationSampler.from_domain(domain)
        batch = sampler.sample(n_col=8192, n_bc_per_region=512)

        # From bounds dict
        sampler = CollocationSampler.from_bounds({"x": (0, 1), "y": (0, 1), "t": (0, 2)})
        batch = sampler.sample(n_col=4096)
    """

    def __init__(
        self,
        *,
        bounds: Dict[str, Tuple[float, float]],
        coord_names: Tuple[str, ...],
        sdf_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        boundary_samplers: Optional[Dict[str, Callable]] = None,
        condition_value_fns: Optional[Dict[str, Callable]] = None,
        fields: Tuple[str, ...] = ("u",),
        strategy: str = "lhs",
        seed: int = 0,
    ):
        self.bounds = bounds
        self.coord_names = tuple(coord_names)
        self.sdf_fn = sdf_fn
        self.boundary_samplers = boundary_samplers or {}
        self.condition_value_fns = condition_value_fns or {}
        self.fields = tuple(fields)
        self.strategy = strategy
        self.seed = seed

        self._bounds_arr = np.array(
            [[bounds.get(c, (0.0, 1.0))[0], bounds.get(c, (0.0, 1.0))[1]] for c in coord_names],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_bounds(
        cls,
        bounds: Dict[str, Tuple[float, float]],
        fields: Tuple[str, ...] = ("u",),
        strategy: str = "lhs",
        seed: int = 0,
    ) -> "CollocationSampler":
        """Create from explicit coord->range dict."""
        coord_names = tuple(bounds.keys())
        return cls(
            bounds=bounds,
            coord_names=coord_names,
            fields=fields,
            strategy=strategy,
            seed=seed,
        )

    @classmethod
    def from_problem_spec(
        cls,
        spec,
        strategy: str = "lhs",
        seed: int = 0,
    ) -> "CollocationSampler":
        """Create from a pinneaple_environment ProblemSpec."""
        bounds = dict(getattr(spec, "domain_bounds", {}))
        coord_names = tuple(spec.coords)
        for c in coord_names:
            if c not in bounds:
                bounds[c] = (0.0, 1.0)

        fields = tuple(spec.fields)

        # Build boundary samplers from conditions
        boundary_samplers = {}
        condition_value_fns = {}

        for cond in spec.conditions:
            name = cond.name
            sel_type = getattr(cond, "selector_type", None)
            sel = getattr(cond, "selector", None)
            vfn = getattr(cond, "value_fn", None)

            if sel_type == "tag" and isinstance(sel, dict):
                tag = sel.get("tag", "boundary")
                from pinneaple_solvers.problem_runner import _sample_boundary_tag

                def _make_sampler(t, b, cn):
                    def _s(n, rng):
                        return _sample_boundary_tag(t, b, cn, n, rng)
                    return _s

                boundary_samplers[name] = _make_sampler(tag, bounds, coord_names)

            elif sel_type == "callable" and callable(sel):
                from pinneaple_solvers.problem_runner import _sample_callable_condition, _make_ctx
                ctx = _make_ctx(bounds)

                def _make_callable_sampler(sfn, b, cn, c):
                    def _s(n, rng):
                        return _sample_callable_condition(sfn, b, cn, n, rng, c)
                    return _s

                boundary_samplers[name] = _make_callable_sampler(sel, bounds, coord_names, _make_ctx(bounds))

            if vfn is not None:
                condition_value_fns[name] = vfn

        return cls(
            bounds=bounds,
            coord_names=coord_names,
            fields=fields,
            boundary_samplers=boundary_samplers,
            condition_value_fns=condition_value_fns,
            strategy=strategy,
            seed=seed,
        )

    @classmethod
    def from_domain(
        cls,
        domain,  # PhysicsDomain2D
        fields: Tuple[str, ...] = ("u", "v", "p"),
        strategy: str = "lhs",
        seed: int = 0,
    ) -> "CollocationSampler":
        """Create from a pinneaple_geom PhysicsDomain2D."""
        bmin = domain.bounds_min
        bmax = domain.bounds_max
        bounds = {"x": (bmin[0], bmax[0]), "y": (bmin[1], bmax[1])}
        coord_names = ("x", "y")

        boundary_samplers = {}
        for region in domain.boundary_regions:
            def _make(dom, rname):
                def _s(n, rng):
                    return dom.sample_boundary_region(rname, n, seed=int(rng.integers(0, 10000)))
                return _s
            boundary_samplers[region.name] = _make(domain, region.name)

        # SDF for interior filtering
        sdf_fn = domain.sdf

        return cls(
            bounds=bounds,
            coord_names=coord_names,
            sdf_fn=sdf_fn,
            boundary_samplers=boundary_samplers,
            fields=fields,
            strategy=strategy,
            seed=seed,
        )

    @classmethod
    def from_mesh(
        cls,
        mesh,  # Mesh2D
        fields: Tuple[str, ...] = ("u",),
        strategy: str = "lhs",
        seed: int = 0,
    ) -> "CollocationSampler":
        """Create from a pinneaple_geom Mesh2D."""
        v = mesh.vertices
        bounds = {"x": (float(v[:, 0].min()), float(v[:, 0].max())),
                  "y": (float(v[:, 1].min()), float(v[:, 1].max()))}

        boundary_samplers = {}
        for region_name, vertex_ids in mesh.boundary_points.items():
            pts = v[vertex_ids].astype(np.float32)
            def _make(p):
                def _s(n, rng):
                    idx = rng.choice(len(p), min(n, len(p)), replace=(n > len(p)))
                    return p[idx]
                return _s
            boundary_samplers[region_name] = _make(pts)

        # Use mesh interior sampler as SDF surrogate
        def _sdf_from_mesh(pts):
            # Simple bbox SDF as fallback
            lo = v.min(axis=0)
            hi = v.max(axis=0)
            return np.maximum(np.max(np.abs(pts - (lo + hi) * 0.5) - (hi - lo) * 0.5, axis=-1), 0.0)

        return cls(
            bounds=bounds,
            coord_names=("x", "y"),
            sdf_fn=_sdf_from_mesh,
            boundary_samplers=boundary_samplers,
            fields=fields,
            strategy=strategy,
            seed=seed,
        )

    # ------------------------------------------------------------------ sampling

    def _interior_points(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n interior points using configured strategy."""
        if self.strategy == "sobol":
            pts = _sample_sobol(self._bounds_arr, n, self.seed)
        elif self.strategy == "lhs":
            pts = _sample_lhs(self._bounds_arr, n, rng)
        else:
            pts = _sample_uniform(self._bounds_arr, n, rng)

        if self.sdf_fn is not None:
            # Rejection to ensure points are inside
            collected = [pts[self.sdf_fn(pts.astype(np.float64)) <= 0]]
            while sum(len(c) for c in collected) < n:
                extra = _sample_uniform(self._bounds_arr, n, rng)
                inside = extra[self.sdf_fn(extra.astype(np.float64)) <= 0]
                if len(inside) > 0:
                    collected.append(inside)
            pts = np.concatenate(collected, axis=0)[:n]

        return pts.astype(np.float32)

    def sample(
        self,
        n_col: Optional[int] = None,
        n_bc: Optional[int] = None,
        n_bc_per_region: Optional[int] = None,
        n_ic: Optional[int] = None,
        seed: Optional[int] = None,
        residual_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate a full PINN training batch.

        Parameters
        ----------
        n_col : number of interior collocation points
        n_bc : total boundary points (split across regions)
        n_bc_per_region : boundary points per region (overrides n_bc)
        n_ic : initial condition points (for time-dependent)
        seed : random seed (overrides instance seed)
        residual_fn : for adaptive strategy, callable (N, D) -> (N,)

        Returns
        -------
        dict with: x_col, x_bc, y_bc, ctx, and optionally x_ic, y_ic
        All tensors are float32 numpy arrays.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        _n_col = n_col or 4096
        _n_bc = n_bc or 1024
        _n_ic = n_ic or 0

        # Interior collocation
        if self.strategy == "adaptive" and residual_fn is not None:
            x_col = _sample_adaptive(self._bounds_arr, _n_col, residual_fn, rng)
        else:
            x_col = self._interior_points(_n_col, rng)

        # Boundary points
        x_bc_parts = []
        y_bc_parts = []
        region_labels = []

        if self.boundary_samplers:
            regions = list(self.boundary_samplers.keys())
            n_per = n_bc_per_region or max(1, _n_bc // len(regions))
            for rname, sampler_fn in self.boundary_samplers.items():
                pts = sampler_fn(n_per, rng)
                if pts is None or len(pts) == 0:
                    continue
                pts = np.asarray(pts, dtype=np.float32)
                x_bc_parts.append(pts)
                region_labels.extend([rname] * len(pts))

                # Target values from condition value_fns
                vfn = self.condition_value_fns.get(rname)
                if vfn is not None:
                    ctx_dict = {"bounds": {c: self.bounds.get(c, (0.0, 1.0)) for c in self.coord_names}}
                    try:
                        yvals = vfn(pts, ctx_dict)
                        y_bc_parts.append(np.asarray(yvals, dtype=np.float32))
                    except Exception:
                        y_bc_parts.append(np.zeros((len(pts), len(self.fields)), dtype=np.float32))
                else:
                    y_bc_parts.append(np.zeros((len(pts), len(self.fields)), dtype=np.float32))
        else:
            # No named regions: sample on all boundaries uniformly
            from pinneaple_solvers.problem_runner import _sample_boundary_tag
            pts = _sample_boundary_tag("boundary", self.bounds, self.coord_names, _n_bc, rng)
            x_bc_parts.append(pts)
            y_bc_parts.append(np.zeros((len(pts), len(self.fields)), dtype=np.float32))

        x_bc = np.concatenate(x_bc_parts, axis=0) if x_bc_parts else np.zeros((0, len(self.coord_names)), dtype=np.float32)
        y_bc = np.concatenate(y_bc_parts, axis=0) if y_bc_parts else np.zeros((0, len(self.fields)), dtype=np.float32)

        # IC points
        x_ic = np.zeros((0, len(self.coord_names)), dtype=np.float32)
        y_ic = np.zeros((0, len(self.fields)), dtype=np.float32)
        if _n_ic > 0 and "t" in self.coord_names:
            from pinneaple_solvers.problem_runner import _sample_boundary_tag
            x_ic = _sample_boundary_tag("ic", self.bounds, self.coord_names, _n_ic, rng)
            y_ic = np.zeros((len(x_ic), len(self.fields)), dtype=np.float32)

        ctx = {
            "bounds": {c: self.bounds.get(c, (0.0, 1.0)) for c in self.coord_names},
            "coord_names": list(self.coord_names),
            "fields": list(self.fields),
            "strategy": self.strategy,
        }

        return {
            "x_col": x_col,
            "x_bc": x_bc,
            "y_bc": y_bc,
            "x_ic": x_ic,
            "y_ic": y_ic,
            "bc_regions": region_labels,
            "ctx": ctx,
        }

    def sample_adaptive(
        self,
        model,
        problem_spec=None,
        n_col: int = 4096,
        n_bc: int = 1024,
        seed: int = 0,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Adaptive sampling using current model residuals.

        Requires a trained model callable (numpy in -> numpy out).
        """
        import torch

        def residual_fn(pts: np.ndarray) -> np.ndarray:
            x_t = torch.from_numpy(pts.astype(np.float32)).to(device)
            x_t.requires_grad_(True)
            with torch.enable_grad():
                out = model(x_t)
                if not isinstance(out, torch.Tensor):
                    for attr in ("y", "pred", "out"):
                        if hasattr(out, attr):
                            out = getattr(out, attr)
                            break
                # Simple proxy: L2 norm of output as residual indicator
                res = out.norm(dim=-1)
            return res.detach().cpu().numpy()

        rng = np.random.default_rng(seed)
        x_col = _sample_adaptive(self._bounds_arr, n_col, residual_fn, rng)

        # Standard boundary sampling
        batch = self.sample(n_col=0, n_bc=n_bc, seed=seed)
        batch["x_col"] = x_col
        return batch
