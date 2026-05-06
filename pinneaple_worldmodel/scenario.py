"""Physics scenario definitions for world model training.

A :class:`PhysicsScenario` fully specifies one physics problem:
PDE kind, grid, time span, parameter ranges, IC/BC types, and which
Pinneaple solver to use.  The :data:`BUILTIN_SCENARIOS` registry provides
ready-made scenarios covering common PDE families.

Quick start::

    from pinneaple_worldmodel import PhysicsScenario, BUILTIN_SCENARIOS

    scenario = BUILTIN_SCENARIOS["heat_2d"]
    # or build a custom one:
    scenario = PhysicsScenario(
        name="my_burgers",
        pde_kind="burgers",
        grid_shape=(128,),
        t_span=(0.0, 1.0),
        n_steps=64,
        domain_bounds=((0.0, 1.0),),
        param_ranges={"nu": (0.001, 0.1)},
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PhysicsScenario
# ---------------------------------------------------------------------------

@dataclass
class PhysicsScenario:
    """Complete specification of a physics simulation problem.

    Parameters
    ----------
    name : str
        Unique identifier (used as dataset directory name).
    pde_kind : str
        PDE family. Built-in values: ``"heat"``, ``"burgers"``, ``"wave"``,
        ``"advection"``, ``"ns2d"``, ``"elasticity"``.
    grid_shape : tuple of int
        Spatial grid dimensions.  1D → ``(Nx,)``, 2D → ``(Nx, Ny)``.
    t_span : (t0, t1)
        Simulation time interval.
    n_steps : int
        Number of discrete time steps in the trajectory.
    domain_bounds : tuple of (lo, hi) pairs
        Spatial domain per dimension, e.g. ``((0,1),(0,1))``.
    param_ranges : dict
        PDE parameter search space.  Each value is ``(lo, hi)`` drawn uniformly.
        Common keys per PDE:

        * heat     : ``alpha`` (diffusivity)
        * burgers  : ``nu`` (viscosity)
        * wave     : ``c`` (wave speed)
        * advection: ``vx``, ``vy`` (advection velocity)
        * ns2d     : ``Re`` (Reynolds number)
    ic_type : str
        Initial condition generator: ``"random_smooth"``, ``"gaussian"``,
        ``"sine"``, ``"step"``, ``"random"``.
    bc_type : str
        Boundary condition: ``"periodic"``, ``"dirichlet_zero"``, ``"neumann_zero"``.
    solver : str
        Which Pinneaple solver to use: ``"fdm"``, ``"lbm"``, ``"fem"``,
        ``"sph"``, ``"builtin"`` (pure-PyTorch fallback, always available).
    n_samples : int
        Default number of random-parameter samples to generate.
    dt : float or None
        Time step size.  ``None`` → computed from ``t_span / n_steps``.
    description : str
        Human-readable description (used in logs and reports).
    tags : list of str
        Free-form tags for filtering (``"fluid"``, ``"2d"``, ``"parabolic"``…).
    """

    name: str
    pde_kind: str
    grid_shape: Tuple[int, ...]
    t_span: Tuple[float, float] = (0.0, 1.0)
    n_steps: int = 32
    domain_bounds: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ic_type: str = "random_smooth"
    bc_type: str = "periodic"
    solver: str = "builtin"
    n_samples: int = 1000
    dt: Optional[float] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.param_ranges:
            self.param_ranges = _default_param_ranges(self.pde_kind)
        if self.dt is None:
            t0, t1 = self.t_span
            self.dt = (t1 - t0) / max(self.n_steps, 1)
        if not self.description:
            self.description = f"{self.pde_kind} on {self.grid_shape} grid"

    @property
    def spatial_dim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.grid_shape)

    @property
    def n_fields(self) -> int:
        """Number of output fields (channels)."""
        return _n_fields(self.pde_kind)

    @property
    def field_names(self) -> List[str]:
        return _field_names(self.pde_kind)

    @property
    def n_params(self) -> int:
        return len(self.param_ranges)


# ---------------------------------------------------------------------------
# PDE metadata helpers
# ---------------------------------------------------------------------------

def _default_param_ranges(pde_kind: str) -> Dict[str, Tuple[float, float]]:
    return {
        "heat":       {"alpha": (0.005, 0.1)},
        "burgers":    {"nu": (0.001, 0.05)},
        "wave":       {"c": (0.5, 2.0)},
        "advection":  {"vx": (0.1, 1.0), "vy": (0.1, 1.0)},
        "ns2d":       {"Re": (100.0, 1000.0)},
        "elasticity": {"E": (1e5, 1e7), "nu": (0.2, 0.45)},
    }.get(pde_kind, {})


def _n_fields(pde_kind: str) -> int:
    return {"heat": 1, "burgers": 1, "wave": 1, "advection": 1,
            "ns2d": 3, "elasticity": 2}.get(pde_kind, 1)


def _field_names(pde_kind: str) -> List[str]:
    return {
        "heat":       ["T"],
        "burgers":    ["u"],
        "wave":       ["u"],
        "advection":  ["phi"],
        "ns2d":       ["u", "v", "p"],
        "elasticity": ["ux", "uy"],
    }.get(pde_kind, ["u"])


# ---------------------------------------------------------------------------
# Built-in scenario registry
# ---------------------------------------------------------------------------

BUILTIN_SCENARIOS: Dict[str, PhysicsScenario] = {
    # ---- 1D ----------------------------------------------------------------
    "burgers_1d": PhysicsScenario(
        name="burgers_1d",
        pde_kind="burgers",
        grid_shape=(256,),
        t_span=(0.0, 1.0),
        n_steps=64,
        domain_bounds=((0.0, 1.0),),
        param_ranges={"nu": (0.001, 0.05)},
        ic_type="sine",
        bc_type="periodic",
        solver="builtin",
        n_samples=2000,
        tags=["1d", "nonlinear", "hyperbolic-parabolic"],
        description="Viscous Burgers: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²",
    ),
    "wave_1d": PhysicsScenario(
        name="wave_1d",
        pde_kind="wave",
        grid_shape=(256,),
        t_span=(0.0, 2.0),
        n_steps=64,
        domain_bounds=((0.0, 1.0),),
        param_ranges={"c": (0.5, 2.0)},
        ic_type="gaussian",
        bc_type="periodic",
        solver="builtin",
        n_samples=2000,
        tags=["1d", "hyperbolic"],
        description="Wave equation: ∂²u/∂t² = c²·∂²u/∂x²",
    ),
    # ---- 2D ----------------------------------------------------------------
    "heat_2d": PhysicsScenario(
        name="heat_2d",
        pde_kind="heat",
        grid_shape=(64, 64),
        t_span=(0.0, 1.0),
        n_steps=32,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        param_ranges={"alpha": (0.005, 0.05)},
        ic_type="random_smooth",
        bc_type="dirichlet_zero",
        solver="builtin",
        n_samples=1000,
        tags=["2d", "parabolic", "diffusion"],
        description="Heat equation: ∂T/∂t = α·∇²T",
    ),
    "advection_2d": PhysicsScenario(
        name="advection_2d",
        pde_kind="advection",
        grid_shape=(64, 64),
        t_span=(0.0, 1.0),
        n_steps=32,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        param_ranges={"vx": (0.2, 1.0), "vy": (0.2, 1.0)},
        ic_type="gaussian",
        bc_type="periodic",
        solver="builtin",
        n_samples=1000,
        tags=["2d", "hyperbolic", "transport"],
        description="Linear advection: ∂φ/∂t + v·∇φ = 0",
    ),
    "ns2d_cavity": PhysicsScenario(
        name="ns2d_cavity",
        pde_kind="ns2d",
        grid_shape=(64, 64),
        t_span=(0.0, 5.0),
        n_steps=50,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        param_ranges={"Re": (100.0, 1000.0)},
        ic_type="random_smooth",
        bc_type="dirichlet_zero",
        solver="builtin",
        n_samples=500,
        tags=["2d", "fluid", "navier-stokes", "incompressible"],
        description="2D lid-driven cavity: incompressible Navier-Stokes",
    ),
    # ---- Multi-scale -------------------------------------------------------
    "heat_multiscale": PhysicsScenario(
        name="heat_multiscale",
        pde_kind="heat",
        grid_shape=(128, 128),
        t_span=(0.0, 2.0),
        n_steps=64,
        domain_bounds=((0.0, 1.0), (0.0, 1.0)),
        param_ranges={"alpha": (0.001, 0.1)},
        ic_type="random_smooth",
        bc_type="periodic",
        solver="builtin",
        n_samples=500,
        tags=["2d", "parabolic", "high-res"],
        description="High-resolution heat equation for fine-tuning stage",
    ),
}
