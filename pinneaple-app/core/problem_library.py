"""Built-in physics problem library — wraps pinneaple_environment presets."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Inline problem definitions (fallback when pinneaple_environment unavailable)
# ---------------------------------------------------------------------------

PROBLEMS: Dict[str, Dict] = {
    "burgers_1d": {
        "name":        "Burgers Equation (1D)",
        "category":    "Fluid Dynamics",
        "description": "1-D viscous Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²",
        "equations":   ["∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²"],
        "domain":      {"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        "params":      {"nu": 0.01},
        "bcs":         ["u(-1,t)=0", "u(1,t)=0"],
        "ics":         ["u(x,0) = -sin(π x)"],
        "dim":         2,
        "tags":        ["pde", "nonlinear", "hyperbolic"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Raissi et al. (2019), PINN paper",
    },
    "heat_2d": {
        "name":        "Heat Equation (2D steady)",
        "category":    "Heat Transfer",
        "description": "2-D steady-state heat conduction: ∇²T = -Q/k",
        "equations":   ["∂²T/∂x² + ∂²T/∂y² = -Q/k"],
        "domain":      {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        "params":      {"k": 1.0, "Q": 1.0},
        "bcs":         ["T=0 on all boundaries"],
        "ics":         [],
        "dim":         2,
        "tags":        ["pde", "elliptic", "linear"],
        "solvers":     ["fdm", "fem", "pinn"],
        "ref":         "Classic benchmark",
    },
    "poisson_2d": {
        "name":        "Poisson Equation (2D)",
        "category":    "Electromagnetics / Structural",
        "description": "∇²u = f(x,y) — prototypical elliptic PDE",
        "equations":   ["∇²u = -2π² sin(πx) sin(πy)"],
        "domain":      {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        "params":      {},
        "bcs":         ["u=0 on ∂Ω"],
        "ics":         [],
        "dim":         2,
        "tags":        ["pde", "elliptic"],
        "solvers":     ["fdm", "fem", "rbf_collocation", "pinn"],
        "ref":         "PINN benchmarks",
    },
    "navier_stokes_2d": {
        "name":        "Navier-Stokes (2D incompressible)",
        "category":    "Fluid Dynamics / CFD",
        "description": "Incompressible N-S: ∂u/∂t + (u·∇)u = -∇p + ν∇²u,  ∇·u=0",
        "equations":   [
            "∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)",
            "∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)",
            "∂u/∂x + ∂v/∂y = 0",
        ],
        "domain":      {"x": (0.0, 2.0), "y": (0.0, 1.0)},
        "params":      {"nu": 0.01, "Re": 100},
        "bcs":         ["no-slip walls", "inlet: u=1", "outlet: p=0"],
        "ics":         ["u=v=0"],
        "dim":         3,
        "tags":        ["pde", "nonlinear", "parabolic", "cfd"],
        "solvers":     ["fvm", "lbm", "pinn"],
        "ref":         "Raissi et al. (2020) — hidden fluid mechanics",
    },
    "wave_1d": {
        "name":        "Wave Equation (1D)",
        "category":    "Acoustics / Structural",
        "description": "∂²u/∂t² = c² ∂²u/∂x²",
        "equations":   ["∂²u/∂t² = c² ∂²u/∂x²"],
        "domain":      {"x": (0.0, 1.0), "t": (0.0, 1.0)},
        "params":      {"c": 1.0},
        "bcs":         ["u(0,t)=0", "u(1,t)=0"],
        "ics":         ["u(x,0) = sin(πx)", "∂u/∂t(x,0) = 0"],
        "dim":         2,
        "tags":        ["pde", "hyperbolic"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Classic benchmark",
    },
    "lid_driven_cavity": {
        "name":        "Lid-Driven Cavity (LBM)",
        "category":    "CFD / LBM",
        "description": "Square cavity with top lid moving at u=1; Re=100–1000",
        "equations":   ["D2Q9 LBM — BGK collision with Zou-He BCs"],
        "domain":      {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        "params":      {"Re": 100, "nx": 64, "ny": 64, "u_in": 0.05},
        "bcs":         ["top lid: u=u_in", "three walls: no-slip"],
        "ics":         [],
        "dim":         2,
        "tags":        ["cfd", "lbm", "benchmark"],
        "solvers":     ["lbm"],
        "ref":         "Ghia et al. (1982)",
    },
    "cylinder_flow": {
        "name":        "Flow Past Cylinder (LBM)",
        "category":    "CFD / LBM",
        "description": "Von Kármán vortex street — classic CFD benchmark",
        "equations":   ["D2Q9 LBM with cylinder bounce-back"],
        "domain":      {"x": (0.0, 4.0), "y": (0.0, 1.0)},
        "params":      {"Re": 200, "nx": 160, "ny": 64, "u_in": 0.05},
        "bcs":         ["velocity inlet", "pressure outlet", "no-slip cylinder"],
        "ics":         [],
        "dim":         2,
        "tags":        ["cfd", "lbm", "wake", "vortex"],
        "solvers":     ["lbm"],
        "ref":         "Schäfer & Turek (1996)",
    },
    "elastic_plate": {
        "name":        "Linear Elasticity (2D)",
        "category":    "Structural Mechanics",
        "description": "Plane-stress elasticity — σ_ij,j + b_i = 0",
        "equations":   ["∂σ_xx/∂x + ∂σ_xy/∂y + bx = 0",
                        "∂σ_xy/∂x + ∂σ_yy/∂y + by = 0"],
        "domain":      {"x": (0.0, 1.0), "y": (0.0, 1.0)},
        "params":      {"E": 1e5, "nu": 0.3},
        "bcs":         ["fixed left edge", "traction on right edge"],
        "ics":         [],
        "dim":         2,
        "tags":        ["structural", "elasticity"],
        "solvers":     ["fem", "pinn"],
        "ref":         "Standard FEM benchmark",
    },
    "reaction_diffusion": {
        "name":        "Reaction-Diffusion (2D)",
        "category":    "Chemical Engineering",
        "description": "Gray-Scott pattern formation: ∂u/∂t = D_u∇²u - uv² + F(1-u)",
        "equations":   ["∂u/∂t = Du ∇²u - uv² + F(1-u)",
                        "∂v/∂t = Dv ∇²v + uv² - (F+k)v"],
        "domain":      {"x": (0.0, 2.5), "y": (0.0, 2.5), "t": (0.0, 10.0)},
        "params":      {"Du": 2e-5, "Dv": 1e-5, "F": 0.04, "k": 0.06},
        "bcs":         ["periodic"],
        "ics":         ["u≈1, v≈0 with small random perturbation"],
        "dim":         3,
        "tags":        ["parabolic", "nonlinear", "chemistry"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Gray & Scott (1984)",
    },
}

CATEGORIES = sorted(set(p["category"] for p in PROBLEMS.values()))


def get_problem(key: str) -> Optional[Dict]:
    return PROBLEMS.get(key)


def list_problems(category: Optional[str] = None) -> List[Tuple[str, Dict]]:
    items = list(PROBLEMS.items())
    if category:
        items = [(k, v) for k, v in items if v["category"] == category]
    return items


def generate_collocation_points(problem: Dict, n_interior: int = 2000,
                                n_boundary: int = 400) -> Dict[str, np.ndarray]:
    """Generate collocation and boundary points for a problem."""
    domain = problem.get("domain", {"x": (0, 1), "y": (0, 1)})
    keys   = list(domain.keys())
    lo     = np.array([domain[k][0] for k in keys])
    hi     = np.array([domain[k][1] for k in keys])

    rng = np.random.default_rng(42)
    interior = rng.uniform(lo, hi, size=(n_interior, len(keys)))

    # Simple boundary sampling (axis-aligned faces)
    boundary_pts = []
    for dim_i in range(len(keys)):
        for val in [lo[dim_i], hi[dim_i]]:
            pts = rng.uniform(lo, hi, size=(n_boundary // (2 * len(keys)), len(keys)))
            pts[:, dim_i] = val
            boundary_pts.append(pts)
    boundary = np.vstack(boundary_pts) if boundary_pts else interior[:n_boundary]

    return {
        "interior":     interior.astype(np.float32),
        "boundary":     boundary.astype(np.float32),
        "coord_names":  keys,
        "domain":       domain,
    }
