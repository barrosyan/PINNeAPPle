"""Generate PINN training datasets from ProblemSpec definitions.

This module bridges ``pinneaple_environment`` problem specifications with the
Arena's dict-batch format expected by the training pipeline.

It handles:
- Collocation point sampling in the problem domain
- Boundary / initial condition point sampling and target evaluation
- Optional reference data generation via registered FDM solvers
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sample_uniform(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(lo, hi, n).astype(np.float32)


def _make_ctx(domain_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Build a ctx dict from domain_bounds, used by condition value_fns."""
    coords = list(domain_bounds.keys())
    bmin = [domain_bounds[c][0] for c in coords]
    bmax = [domain_bounds[c][1] for c in coords]
    return {
        "bounds": {
            "coords": coords,
            "min": bmin,
            "max": bmax,
            **{c: domain_bounds[c] for c in coords},
        }
    }


def _sample_collocation(
    domain_bounds: Dict[str, Tuple[float, float]],
    coord_names: Tuple[str, ...],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n points uniformly in the rectangular domain."""
    dims = len(coord_names)
    pts = np.zeros((n, dims), dtype=np.float32)
    for i, c in enumerate(coord_names):
        lo, hi = domain_bounds.get(c, (0.0, 1.0))
        pts[:, i] = _sample_uniform(lo, hi, n, rng)
    return pts


def _sample_boundary_tag(
    tag: str,
    domain_bounds: Dict[str, Tuple[float, float]],
    coord_names: Tuple[str, ...],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n points on the boundary region identified by tag.

    Supported tags for rectangular domains:
    - "boundary" : all boundary edges equally
    - "inlet"    : x = x_min face
    - "outlet"   : x = x_max face
    - "walls"    : y = y_min and y = y_max faces (2D)
    - "left", "right", "top", "bottom" : individual edges

    For 1D+time:
    - "ic" or "initial" : t = t_min
    """
    dims = len(coord_names)
    coords = list(coord_names)

    # Determine spatial and time axes
    spatial = [c for c in coords if c != "t"]
    has_time = "t" in coords

    def _face(fixed_coord: str, fixed_val: float, n_pts: int) -> np.ndarray:
        pts = np.zeros((n_pts, dims), dtype=np.float32)
        for i, c in enumerate(coords):
            if c == fixed_coord:
                pts[:, i] = fixed_val
            else:
                lo, hi = domain_bounds.get(c, (0.0, 1.0))
                pts[:, i] = _sample_uniform(lo, hi, n_pts, rng)
        return pts

    tag_lower = tag.lower()

    if tag_lower in ("ic", "initial"):
        if "t" not in coords:
            return np.zeros((0, dims), dtype=np.float32)
        t_lo, _ = domain_bounds.get("t", (0.0, 1.0))
        return _face("t", t_lo, n)

    if tag_lower == "inlet" or tag_lower == "left":
        x_coord = spatial[0] if spatial else coords[0]
        lo, _ = domain_bounds.get(x_coord, (0.0, 1.0))
        return _face(x_coord, lo, n)

    if tag_lower == "outlet" or tag_lower == "right":
        x_coord = spatial[0] if spatial else coords[0]
        _, hi = domain_bounds.get(x_coord, (0.0, 1.0))
        return _face(x_coord, hi, n)

    if tag_lower in ("walls", "top_bottom") and len(spatial) >= 2:
        y_coord = spatial[1]
        lo, hi = domain_bounds.get(y_coord, (0.0, 1.0))
        pts_lo = _face(y_coord, lo, n // 2)
        pts_hi = _face(y_coord, hi, n - n // 2)
        return np.concatenate([pts_lo, pts_hi], axis=0)

    if tag_lower == "bottom" and len(spatial) >= 2:
        y_coord = spatial[1]
        lo, _ = domain_bounds.get(y_coord, (0.0, 1.0))
        return _face(y_coord, lo, n)

    if tag_lower == "top" and len(spatial) >= 2:
        y_coord = spatial[1]
        _, hi = domain_bounds.get(y_coord, (0.0, 1.0))
        return _face(y_coord, hi, n)

    if tag_lower == "boundary":
        # All boundary faces combined
        n_each = max(1, n // (2 * len(spatial) + (1 if has_time else 0)))
        parts = []
        x_coord = spatial[0] if spatial else coords[0]
        lo_x, hi_x = domain_bounds.get(x_coord, (0.0, 1.0))
        parts.append(_face(x_coord, lo_x, n_each))
        parts.append(_face(x_coord, hi_x, n_each))
        if len(spatial) >= 2:
            y_coord = spatial[1]
            lo_y, hi_y = domain_bounds.get(y_coord, (0.0, 1.0))
            parts.append(_face(y_coord, lo_y, n_each))
            parts.append(_face(y_coord, hi_y, n_each))
        if has_time:
            t_lo, _ = domain_bounds.get("t", (0.0, 1.0))
            parts.append(_face("t", t_lo, n_each))
        combined = np.concatenate(parts, axis=0)
        # Resample to exactly n
        idx = rng.choice(len(combined), n, replace=(len(combined) < n))
        return combined[idx]

    # Fallback: random interior (should not happen for well-defined tags)
    return _sample_collocation(domain_bounds, coord_names, n, rng)


def _sample_callable_condition(
    selector_fn,
    domain_bounds: Dict[str, Tuple[float, float]],
    coord_names: Tuple[str, ...],
    n: int,
    rng: np.random.Generator,
    ctx: Dict[str, Any],
    oversample: int = 10,
) -> np.ndarray:
    """Sample points that satisfy a callable selector by rejection sampling."""
    dims = len(coord_names)
    collected = []
    total_needed = n
    attempts = 0
    max_attempts = 20

    while len(collected) < total_needed and attempts < max_attempts:
        candidates = _sample_collocation(domain_bounds, coord_names, total_needed * oversample, rng)
        try:
            mask = selector_fn(candidates, ctx)
        except Exception:
            mask = np.ones(len(candidates), dtype=bool)
        selected = candidates[mask]
        if len(selected) > 0:
            collected.append(selected)
        attempts += 1

    if not collected:
        return np.zeros((n, dims), dtype=np.float32)

    all_pts = np.concatenate(collected, axis=0)
    if len(all_pts) >= n:
        idx = rng.choice(len(all_pts), n, replace=False)
        return all_pts[idx]
    # pad by repeating
    idx = rng.choice(len(all_pts), n, replace=True)
    return all_pts[idx]


def generate_pinn_dataset(
    problem_spec: Any,
    *,
    n_col: Optional[int] = None,
    n_bc: Optional[int] = None,
    n_ic: Optional[int] = None,
    n_data: int = 0,
    seed: int = 0,
    solver_cfg_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Generate a PINN training dataset from a ProblemSpec.

    Samples collocation points in the problem domain and boundary/initial
    condition points based on the conditions defined in the spec.

    Parameters
    ----------
    problem_spec : ProblemSpec from pinneaple_environment
    n_col : number of collocation points (defaults to sample_defaults["n_col"])
    n_bc : number of boundary condition points per condition
    n_ic : number of initial condition points
    n_data : number of reference data points (requires solver to run)
    seed : random seed
    solver_cfg_override : override solver_spec params

    Returns
    -------
    dict with keys: x_col, x_bc, y_bc, x_ic, y_ic, (x_data, y_data), ctx
    All values are float32 numpy arrays.
    """
    rng = np.random.default_rng(seed)

    domain_bounds: Dict[str, Tuple[float, float]] = dict(getattr(problem_spec, "domain_bounds", {}))
    coord_names: Tuple[str, ...] = tuple(problem_spec.coords)
    fields: Tuple[str, ...] = tuple(problem_spec.fields)
    conditions = problem_spec.conditions

    # Fill in missing bounds with defaults [0, 1]
    for c in coord_names:
        if c not in domain_bounds:
            domain_bounds[c] = (0.0, 1.0)

    ctx = _make_ctx(domain_bounds)

    # Defaults for sampling counts
    sample_defaults = dict(getattr(problem_spec, "sample_defaults", {}))
    if n_col is None:
        n_col = int(sample_defaults.get("n_col", 10_000))
    if n_bc is None:
        n_bc = int(sample_defaults.get("n_bc", 2_000))
    if n_ic is None:
        n_ic = int(sample_defaults.get("n_ic", 2_000))

    # 1. Collocation points
    x_col = _sample_collocation(domain_bounds, coord_names, n_col, rng)

    # 2. Boundary / initial condition points
    x_bc_list: List[np.ndarray] = []
    y_bc_list: List[np.ndarray] = []
    x_ic_list: List[np.ndarray] = []
    y_ic_list: List[np.ndarray] = []

    for cond in conditions:
        is_ic = getattr(cond, "kind", "") == "initial"
        target_list_x = x_ic_list if is_ic else x_bc_list
        target_list_y = y_ic_list if is_ic else y_bc_list
        n_pts = n_ic if is_ic else n_bc

        # Sample points on the condition's region
        sel_type = getattr(cond, "selector_type", None)
        sel = getattr(cond, "selector", None)
        pts: np.ndarray

        if sel_type == "tag" and isinstance(sel, dict):
            tag = sel.get("tag", "boundary")
            pts = _sample_boundary_tag(tag, domain_bounds, coord_names, n_pts, rng)
        elif sel_type == "callable" and callable(sel):
            pts = _sample_callable_condition(sel, domain_bounds, coord_names, n_pts, rng, ctx)
        elif is_ic:
            # Default IC: t = t_min
            pts = _sample_boundary_tag("ic", domain_bounds, coord_names, n_pts, rng)
        else:
            pts = _sample_boundary_tag("boundary", domain_bounds, coord_names, n_pts, rng)

        if len(pts) == 0:
            continue

        # Evaluate target values
        value_fn = getattr(cond, "value_fn", None)
        if value_fn is not None:
            try:
                targets = value_fn(pts, ctx)
                targets = np.asarray(targets, dtype=np.float32)
            except Exception:
                n_fields_cond = len(getattr(cond, "fields", fields))
                targets = np.zeros((len(pts), n_fields_cond), dtype=np.float32)
        else:
            n_fields_cond = len(getattr(cond, "fields", fields))
            targets = np.zeros((len(pts), n_fields_cond), dtype=np.float32)

        target_list_x.append(pts)
        target_list_y.append(targets)

    def _concat_or_empty(lst: List[np.ndarray], ncols: int) -> np.ndarray:
        if lst:
            return np.concatenate(lst, axis=0).astype(np.float32)
        return np.zeros((0, ncols), dtype=np.float32)

    n_dims = len(coord_names)
    n_fields_out = len(fields)

    x_bc = _concat_or_empty(x_bc_list, n_dims)
    y_bc = _concat_or_empty(y_bc_list, n_fields_out)
    x_ic = _concat_or_empty(x_ic_list, n_dims)
    y_ic = _concat_or_empty(y_ic_list, n_fields_out)

    batch: Dict[str, Any] = {
        "x_col": x_col,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "ctx": ctx,
    }

    # Optional reference data from solver
    if n_data > 0:
        try:
            ref = _run_reference_solver(problem_spec, solver_cfg_override, rng)
            if ref is not None:
                x_data, y_data = _sample_reference(ref, n_data, coord_names, fields, rng)
                batch["x_data"] = x_data
                batch["y_data"] = y_data
        except Exception:
            pass  # solver reference is optional

    return batch


def _run_reference_solver(
    problem_spec: Any,
    solver_cfg_override: Optional[Dict[str, Any]],
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    """Run the problem's associated solver and return a reference field dict.

    Returns dict with keys: "coords" (dict of coord arrays) and field arrays,
    or None if solver not available.
    """
    spec_dict = dict(getattr(problem_spec, "solver_spec", {}))
    if solver_cfg_override:
        spec_dict = {**spec_dict, **solver_cfg_override}

    solver_name = spec_dict.get("name", "")
    method = spec_dict.get("method", "")
    params = dict(spec_dict.get("params", {}))

    if not solver_name:
        return None

    pde_kind = problem_spec.pde.kind.lower()

    # Dispatch to built-in reference solvers
    if "burgers" in pde_kind and method == "burgers_1d":
        return _solve_burgers_1d(problem_spec, params, rng)
    if ("laplace" in pde_kind or "poisson" in pde_kind) and "poisson" in method:
        return _solve_poisson_2d(problem_spec, params, rng)

    return None


def _solve_burgers_1d(
    problem_spec: Any,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simple FTCS (explicit) solver for viscous Burgers 1D.

    u_t + u*u_x = nu * u_xx
    IC: u(x,0) = -sin(pi*x),  BC: u(-1,t) = u(1,t) = 0
    """
    domain_bounds = dict(getattr(problem_spec, "domain_bounds", {"x": (-1.0, 1.0), "t": (0.0, 1.0)}))
    nu = float(params.get("nu", problem_spec.pde.params.get("nu", 0.01)))
    nx = int(params.get("nx", 256))
    nt = int(params.get("nt", 256))
    x_lo, x_hi = domain_bounds.get("x", (-1.0, 1.0))
    t_lo, t_hi = domain_bounds.get("t", (0.0, 1.0))

    x = np.linspace(x_lo, x_hi, nx, dtype=np.float64)
    dx = (x_hi - x_lo) / (nx - 1)
    dt = (t_hi - t_lo) / nt
    t = np.linspace(t_lo, t_hi, nt + 1, dtype=np.float64)

    u = -np.sin(np.pi * x)  # IC
    U = [u.copy()]

    for _ in range(nt):
        u_new = u.copy()
        # central difference for diffusion, upwind for convection
        u_new[1:-1] = (
            u[1:-1]
            - dt / (2 * dx) * u[1:-1] * (u[2:] - u[:-2])
            + nu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        )
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new
        U.append(u.copy())

    U_arr = np.stack(U, axis=1)  # (nx, nt+1)
    return {
        "coords": {"x": x.astype(np.float32), "t": t.astype(np.float32)},
        "u": U_arr.astype(np.float32),
    }


def _solve_poisson_2d(
    problem_spec: Any,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Jacobi iterative solver for Laplace equation (nabla^2 u = 0) with u=0 BC."""
    domain_bounds = dict(getattr(problem_spec, "domain_bounds", {"x": (0.0, 1.0), "y": (0.0, 1.0)}))
    nx = int(params.get("nx", 64))
    ny = int(params.get("ny", 64))
    iters = int(params.get("iters", 2000))
    x_lo, x_hi = domain_bounds.get("x", (0.0, 1.0))
    y_lo, y_hi = domain_bounds.get("y", (0.0, 1.0))

    x = np.linspace(x_lo, x_hi, nx, dtype=np.float64)
    y = np.linspace(y_lo, y_hi, ny, dtype=np.float64)
    u = np.zeros((nx, ny), dtype=np.float64)

    # Example: u=sin(pi*x)*sin(pi*y) BC as a manufactured solution
    XX, YY = np.meshgrid(x, y, indexing="ij")
    # Dirichlet BC = 0 (homogeneous) -- so solution is trivially 0.
    # For a nontrivial case, set interior source f = -2*pi^2*sin(pi*x)*sin(pi*y)
    # giving u = sin(pi*x)*sin(pi*y)
    dx = (x_hi - x_lo) / (nx - 1)
    dy = (y_hi - y_lo) / (ny - 1)
    f = -2 * np.pi**2 * np.sin(np.pi * XX) * np.sin(np.pi * YY)

    for _ in range(iters):
        u[1:-1, 1:-1] = (
            (u[1:-1, 2:] + u[1:-1, :-2]) * dy**2
            + (u[2:, 1:-1] + u[:-2, 1:-1]) * dx**2
            - f[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

    return {
        "coords": {"x": x.astype(np.float32), "y": y.astype(np.float32)},
        "u": u.astype(np.float32),
    }


def _sample_reference(
    ref: Dict[str, np.ndarray],
    n: int,
    coord_names: Tuple[str, ...],
    fields: Tuple[str, ...],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n (x, y) pairs from a reference solution grid."""
    coords_dict = ref["coords"]
    coord_keys = list(coords_dict.keys())

    # Build flat arrays of coordinates and field values
    grids = [coords_dict[c] for c in coord_keys if c in coord_names]
    if len(grids) == 2:
        G1, G2 = np.meshgrid(grids[0], grids[1], indexing="ij")
        flat_coords = {coord_keys[0]: G1.ravel(), coord_keys[1]: G2.ravel()}
        total = G1.size
    elif len(grids) == 1:
        flat_coords = {coord_keys[0]: grids[0].ravel()}
        total = grids[0].size
    else:
        return np.zeros((0, len(coord_names)), dtype=np.float32), np.zeros((0, len(fields)), dtype=np.float32)

    n_eff = min(n, total)
    idx = rng.choice(total, n_eff, replace=False)

    x_data = np.zeros((n_eff, len(coord_names)), dtype=np.float32)
    for i, c in enumerate(coord_names):
        if c in flat_coords:
            x_data[:, i] = flat_coords[c][idx]

    y_data = np.zeros((n_eff, len(fields)), dtype=np.float32)
    for i, fname in enumerate(fields):
        if fname in ref:
            arr = ref[fname]
            if arr.ndim == 2:
                y_data[:, i] = arr.ravel()[idx]
            elif arr.ndim == 1:
                y_data[:, i] = arr[idx]

    return x_data, y_data
