"""
REBOUND N-body gravitational integrator.

Wraps `rebound.Simulation` (Rein & Liu 2012) as a PINNeAPPle numerical
solver: given a list of massive (and optionally massless test-particle)
bodies specified either by Cartesian state (mass/position/velocity) or by
classical orbital elements about a chosen primary, this builds a real
`rebound.Simulation`, integrates it forward through a caller-supplied set
of output times using a real REBOUND integrator (IAS15 by default -- a
15th-order adaptive Gauss-Radau integrator with machine-precision energy
conservation for well-resolved orbits), and returns the resulting
per-body, per-output-time trajectories (positions and velocities) wrapped
in a `SolverOutput`.

This complements the closed-form/perturbative analytical orbital-mechanics
presets in `pinneapple_physics/pde_environment/presets/astrophysics.py`
(two-body Kepler, Clohessy-Wiltshire relative motion, J2 secular
perturbation, and the planar circular restricted three-body problem
(CR3BP) with its Lagrange-point equilibria): those are all reduced,
analytically or semi-analytically tractable models, whereas this solver
integrates the *real, unrestricted* N-body gravitational equations of
motion for an arbitrary number of massive bodies -- e.g. it can reproduce
the same CR3BP Lagrange-point equilibria as a genuine full N-body
simulation (two massive primaries + a massless test particle), rather than
assuming the restricted-three-body reduction from the outset.

Reference
---------
Rein, H., Liu, S.-F. (2012). "REBOUND: An open-source multi-purpose
N-body code for collisional dynamics." Astronomy & Astrophysics, 537,
A128. https://doi.org/10.1051/0004-6361/201118085
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

try:
    import rebound  # type: ignore
    _HAS_REBOUND = True
    _REBOUND_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - exercised only when rebound is absent
    rebound = None  # type: ignore
    _HAS_REBOUND = False
    _REBOUND_IMPORT_ERROR = str(_e)


def _require_rebound() -> None:
    if not _HAS_REBOUND:
        raise ImportError(
            "NBodySolver requires the 'rebound' package (Rein & Liu 2012, "
            "A&A 537, A128), which is not installed or failed to import "
            f"({_REBOUND_IMPORT_ERROR}). Install it with `pip install rebound` "
            "(a real, standard C-extension-backed PyPI package -- "
            "https://github.com/hannorein/rebound)."
        )


@SolverRegistry.register(
    name="nbody_rebound",
    family="pde",
    description=(
        "REBOUND (Rein & Liu 2012) N-body gravitational integrator -- IAS15/WHFast/"
        "etc. wrapped for arbitrary Cartesian or orbital-element initial conditions."
    ),
    tags=["astrophysics", "orbital_mechanics", "n_body"],
)
class NBodySolver(SolverBase):
    """
    Thin, real wrapper around `rebound.Simulation`.

    Parameters
    ----------
    bodies : list[dict]
        One dict per body. Each dict must supply a mass ``m`` (use ``m=0``
        for a massless test particle) and either:
          - a Cartesian state: ``x, y, z, vx, vy, vz`` (any omitted key
            defaults to 0.0), or
          - classical orbital elements about the *primary* (by convention,
            the first body added, i.e. ``bodies[0]``): ``a`` (semi-major
            axis) and optionally ``e, inc, Omega, omega, f`` (REBOUND's
            ``add(primary=..., m=..., a=..., e=..., ...)`` form -- see
            `rebound.Simulation.add`).
        A body is treated as "elements" form iff it contains the key
        ``"a"`` and does NOT contain ``"x"``; otherwise it is treated as
        Cartesian form.
    integrator : str
        Any REBOUND integrator name (default ``"ias15"``, REBOUND's
        adaptive high-precision Gauss-Radau15 integrator -- appropriate
        for close encounters and long-term energy-conservation checks;
        other options include ``"whfast"`` for fast symplectic
        long-term integrations of well-separated orbits, ``"leapfrog"``,
        etc; REBOUND validates the name itself).
    G : float
        Gravitational constant in the simulation's chosen unit system
        (default 1.0, i.e. dimensionless units as used by
        ``cr3bp_planar_synodic`` in the astrophysics presets; set to
        ``6.674e-20`` for km^3/(kg s^2)-style physical units, etc.).
    dt : float, optional
        Initial timestep hint. IAS15 adapts its own timestep
        automatically; for symplectic integrators (e.g. WHFast) this
        should be set explicitly to a small fraction of the shortest
        orbital period.
    move_to_com : bool
        If True (default), shift to the centre-of-mass frame right after
        adding all bodies (``rebound.Simulation.move_to_com()``) so the
        barycentre stays fixed at the origin -- standard practice before
        integrating a self-gravitating system.
    """

    def __init__(
        self,
        bodies: List[Dict[str, float]],
        integrator: str = "ias15",
        G: float = 1.0,
        dt: Optional[float] = None,
        move_to_com: bool = True,
    ):
        _require_rebound()
        super().__init__()
        if not bodies:
            raise ValueError("NBodySolver requires at least one body.")
        self.body_specs = [dict(b) for b in bodies]
        self.integrator = str(integrator)
        self.G = float(G)
        self.dt = dt
        self.move_to_com = bool(move_to_com)
        self.n_bodies = len(self.body_specs)

    # ------------------------------------------------------------------
    def _build_simulation(self) -> "rebound.Simulation":
        """Construct a fresh `rebound.Simulation` from `self.body_specs`."""
        sim = rebound.Simulation()
        sim.G = self.G
        sim.integrator = self.integrator
        if self.dt is not None:
            sim.dt = float(self.dt)

        for i, spec in enumerate(self.body_specs):
            m = float(spec.get("m", 0.0))
            if "a" in spec and "x" not in spec:
                kwargs = {k: float(v) for k, v in spec.items() if k != "m"}
                if i == 0:
                    # First body cannot be given orbital elements (no primary yet).
                    raise ValueError(
                        "bodies[0] (the primary) must be specified by Cartesian "
                        "state (x,y,z,vx,vy,vz), not orbital elements."
                    )
                sim.add(m=m, primary=sim.particles[0], **kwargs)
            else:
                sim.add(
                    m=m,
                    x=float(spec.get("x", 0.0)),
                    y=float(spec.get("y", 0.0)),
                    z=float(spec.get("z", 0.0)),
                    vx=float(spec.get("vx", 0.0)),
                    vy=float(spec.get("vy", 0.0)),
                    vz=float(spec.get("vz", 0.0)),
                )

        if self.move_to_com:
            sim.move_to_com()
        return sim

    # ------------------------------------------------------------------
    def forward(
        self,
        t_eval: Optional[Sequence[float]] = None,
        *,
        t_end: Optional[float] = None,
        n_steps: int = 100,
        exact_finish_time: bool = True,
    ) -> SolverOutput:
        """
        Integrate the N-body system forward in time.

        Parameters
        ----------
        t_eval : sequence of float, optional
            Explicit output times (must be non-decreasing, start >= 0).
            If given, `t_end`/`n_steps` are ignored.
        t_end : float, optional
            If `t_eval` is not given, integrate from t=0 to `t_end` and
            record `n_steps + 1` equally spaced output times (including
            t=0).
        n_steps : int
            Number of integration/output intervals when `t_eval` is not
            given (ignored otherwise).
        exact_finish_time : bool
            Passed through to `rebound.Simulation.integrate` at each
            output time (REBOUND's own option controlling whether it
            integrates exactly to the requested time or stops at the
            nearest completed step).

        Returns
        -------
        SolverOutput
          result        : positions tensor, shape (n_times, n_bodies, 3)
          extras['t']          : output times, shape (n_times,)
          extras['positions']  : (n_times, n_bodies, 3) torch.Tensor
          extras['velocities'] : (n_times, n_bodies, 3) torch.Tensor
          extras['masses']     : (n_bodies,) torch.Tensor
          extras['energy']     : (n_times,) torch.Tensor -- total energy
                                  (`rebound.Simulation.energy()`) at each
                                  output time, for conservation checks
          extras['final_sim']  : the underlying `rebound.Simulation`
                                  after integration (for inspection; not
                                  a tensor)
        """
        if t_eval is not None:
            times = np.asarray(list(t_eval), dtype=np.float64)
        else:
            if t_end is None:
                raise ValueError("Must supply either t_eval or t_end.")
            times = np.linspace(0.0, float(t_end), int(n_steps) + 1)

        sim = self._build_simulation()
        sim.exact_finish_time = 1 if exact_finish_time else 0

        n_t = len(times)
        positions = np.zeros((n_t, self.n_bodies, 3), dtype=np.float64)
        velocities = np.zeros((n_t, self.n_bodies, 3), dtype=np.float64)
        energies = np.zeros((n_t,), dtype=np.float64)
        masses = np.array([p.m for p in sim.particles], dtype=np.float64)

        for ti, t in enumerate(times):
            if t > 0.0:
                sim.integrate(float(t))
            for bi, p in enumerate(sim.particles):
                positions[ti, bi] = (p.x, p.y, p.z)
                velocities[ti, bi] = (p.vx, p.vy, p.vz)
            energies[ti] = sim.energy()

        pos_t = torch.as_tensor(positions, dtype=torch.float64)
        vel_t = torch.as_tensor(velocities, dtype=torch.float64)

        return SolverOutput(
            result=pos_t,
            losses={},
            extras={
                "t": torch.as_tensor(times, dtype=torch.float64),
                "positions": pos_t,
                "velocities": vel_t,
                "masses": torch.as_tensor(masses, dtype=torch.float64),
                "energy": torch.as_tensor(energies, dtype=torch.float64),
                "integrator": self.integrator,
                "G": self.G,
                "final_sim": sim,
            },
        )

    # ------------------------------------------------------------------
    def solve(self, *args, **kwargs) -> SolverOutput:
        """Alias for `forward`, matching the `solve(...)` convention used
        elsewhere in the numerical-solvers package alongside `forward`."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    def step(self, t: float, exact_finish_time: bool = True) -> SolverOutput:
        """Integrate a fresh simulation from t=0 to a single time `t` and
        return the resulting one-snapshot `SolverOutput` (convenience
        wrapper around `forward(t_eval=[t])`)."""
        return self.forward(t_eval=[float(t)], exact_finish_time=exact_finish_time)
