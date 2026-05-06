"""Data generation pipeline — runs a solver to produce synthetic training data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    solver_key: str = "fdm_2d_generic"   # key from problem_registry.SOLVER_MAP
    n_snapshots: int = 10                 # trajectories / time snapshots
    grid_resolution: int = 32            # per-dimension for FDM solvers
    t_end: float = 1.0
    use_solver: bool = True              # False = collocation only (PINN)


@dataclass
class DataBundle:
    """Output of the data pipeline, consumed by ExperimentRunner."""
    x_col: np.ndarray      # interior collocation points (N, D)
    x_bnd: np.ndarray      # boundary points            (M, D)
    x_ic:  Optional[np.ndarray] = None    # IC points  (K, D)
    u_ref: Optional[np.ndarray] = None    # reference field (N, C) from solver
    solver_outputs: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def run_data_pipeline(
    problem,                         # ProblemDefinition
    collocation_cfg,                 # CollocationConfig
    data_cfg: DataConfig,
    *,
    device: str = "cpu",
    verbose: bool = True,
) -> DataBundle:
    """Execute data generation: (optionally) run solver + sample collocation pts."""
    from .collocation import generate_points

    if verbose:
        print(f"[DataPipeline] Generating collocation points  "
              f"(strategy={collocation_cfg.strategy}, "
              f"n_int={collocation_cfg.n_interior})")

    pts = generate_points(
        collocation_cfg,
        problem.domain_bounds,
        include_time=problem.is_time_dependent,
    )

    solver_outputs = []
    u_ref = None

    if data_cfg.use_solver and data_cfg.solver_key not in ("uniform", "none", ""):
        if verbose:
            print(f"[DataPipeline] Running solver: {data_cfg.solver_key}")
        solver_outputs, u_ref = _run_solver(
            data_cfg, problem, pts["interior"]
        )

    return DataBundle(
        x_col=pts["interior"],
        x_bnd=pts["boundary"],
        x_ic=pts.get("initial"),
        u_ref=u_ref,
        solver_outputs=solver_outputs,
    )


def _run_solver(
    cfg: DataConfig,
    problem,
    x_interior: np.ndarray,
) -> tuple:
    """Dispatch to the appropriate solver and return (outputs, ref_field)."""
    from .problem_registry import SOLVER_MAP

    solver_info = SOLVER_MAP.get(cfg.solver_key, {})
    cls_name = solver_info.get("class")

    try:
        import pinneaple_simulation.numerical_solvers as _sol
        cls = getattr(_sol, cls_name, None)
        if cls is None:
            return [], None
    except ImportError:
        return [], None

    try:
        dim = solver_info.get("dim", problem.dim)
        bounds = problem.domain_bounds
        coord_names = list(bounds.keys())

        if cls_name == "HeatConduction3D":
            from pinneaple_simulation.numerical_solvers import HeatConfig3D, HeatConduction3D
            nx = cfg.grid_resolution
            b = list(bounds.values())
            solver = HeatConduction3D(HeatConfig3D(
                nx=nx, ny=nx, nz=nx if dim >= 3 else 1,
                lx=b[0][1]-b[0][0],
                ly=b[1][1]-b[1][0] if len(b) > 1 else 1.0,
                lz=b[2][1]-b[2][0] if len(b) > 2 else 1.0,
                t_end=cfg.t_end, n_steps=cfg.n_snapshots,
            ))
            out = solver.run()
            u_ref = _interpolate_solver_to_pts(out, x_interior)
            return [out], u_ref

        elif cls_name == "NavierStokes3D":
            from pinneaple_simulation.numerical_solvers import NavierStokesConfig3D, NavierStokes3D
            nx = cfg.grid_resolution
            b = list(bounds.values())
            solver = NavierStokes3D(NavierStokesConfig3D(
                nx=nx, ny=nx, nz=nx if dim >= 3 else 1,
                lx=b[0][1]-b[0][0],
                ly=b[1][1]-b[1][0] if len(b) > 1 else 1.0,
                lz=b[2][1]-b[2][0] if len(b) > 2 else 1.0,
                t_end=cfg.t_end, n_steps=cfg.n_snapshots,
            ))
            out = solver.run()
            u_ref = _interpolate_solver_to_pts(out, x_interior)
            return [out], u_ref

        elif cls_name == "LidDrivenCavitySolver3D":
            from pinneaple_simulation.numerical_solvers import LidDrivenCavityConfig3D, LidDrivenCavitySolver3D
            solver = LidDrivenCavitySolver3D(LidDrivenCavityConfig3D(
                nx=cfg.grid_resolution, ny=cfg.grid_resolution,
                t_end=cfg.t_end, n_steps=cfg.n_snapshots,
            ))
            out = solver.run()
            u_ref = _interpolate_solver_to_pts(out, x_interior)
            return [out], u_ref

        else:
            return [], None

    except Exception as e:
        print(f"[DataPipeline] Solver failed ({e}), continuing without reference data.")
        return [], None


def _interpolate_solver_to_pts(solver_out, x_pts: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort interpolation of solver field onto collocation points."""
    try:
        from scipy.interpolate import RegularGridInterpolator
        # SolverOutput3D exposes .fields dict with numpy arrays
        if not hasattr(solver_out, "fields") or not solver_out.fields:
            return None
        first_field = next(iter(solver_out.fields.values()))
        # shape: (nx, ny, nz) or (nx, ny)
        ndim = first_field.ndim
        if ndim == 2:
            nx, ny = first_field.shape
            xs = np.linspace(0, 1, nx)
            ys = np.linspace(0, 1, ny)
            interp = RegularGridInterpolator((xs, ys), first_field, method="linear",
                                             bounds_error=False, fill_value=0.0)
            pts_clipped = np.clip(x_pts[:, :2], 0, 1)
            return interp(pts_clipped).reshape(-1, 1).astype(np.float32)
        return None
    except Exception:
        return None
