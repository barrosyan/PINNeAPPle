"""JAX-CFD adapter — real differentiable Navier-Stokes via Google's jax-cfd.

Wraps `jax-cfd <https://github.com/google/jax-cfd>`_, Google Research's
open-source differentiable computational-fluid-dynamics library built on
JAX, as a registered PINNeAPPle numerical solver. ``jax_cfd.base`` provides
a finite-volume, MAC-grid incompressible Navier-Stokes solver (semi-implicit
time stepping: explicit advection + implicit/explicit diffusion + a spectral
pressure projection that enforces the divergence-free constraint) that runs
end-to-end in JAX, so it is (in principle) differentiable and JIT/vmap-able.
This module drives the real installed library — it does not reimplement the
solver.

References
----------
Kochkov, D., Smith, J. A., Alieva, A., Wang, Q., Brenner, M. P., & Hoyer, S.
(2021). "Machine learning-accelerated computational fluid dynamics."
Proceedings of the National Academy of Sciences, 118(21).

Dresdner, G., Kochkov, D., Norgaard, P., Zepeda-Nunez, L., Smith, J. A.,
Brenner, M. P., & Hoyer, S. (2022). "Learning to correct spectral methods
for simulating turbulent flows." arXiv:2207.00556.

Project: https://github.com/google/jax-cfd (Apache-2.0, Google LLC).

Optional dependency
--------------------
``jax_cfd`` (and its own ``jax``/``jaxlib`` dependency) is an optional,
install-on-demand dependency of PINNeAPPle, following the same
try/except-ImportError convention used by
``pinneapple_tools.compute_backends.jax_backend`` for plain JAX: importing
this module always succeeds, but constructing/running :class:`JAXCFDSolver`
raises a clear ``ImportError`` (with the install command) if ``jax_cfd`` is
not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


# ===========================================================================
# Optional-dependency handling (mirrors jax_backend.jax_available())
# ===========================================================================

def jax_cfd_available() -> bool:
    """Return ``True`` if both ``jax`` and ``jax_cfd`` can be imported.

    As a side effect, enables JAX's 64-bit precision (``jax_enable_x64``),
    matching :func:`pinneapple_tools.compute_backends.jax_backend.jax_available`
    so a torch-float64 initial condition round-tripped through JAX does not
    get silently downcast to float32.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax_cfd.base  # noqa: F401
        return True
    except ImportError:
        return False


def _require_jax_cfd() -> None:
    if not jax_cfd_available():
        raise ImportError(
            "jax-cfd is not installed. Install with: pip install jax jaxlib jax-cfd "
            "(see https://github.com/google/jax-cfd)."
        )


# ===========================================================================
# Built-in initial conditions
# ===========================================================================

def taylor_green_velocity_fns() -> Tuple[Callable, Callable]:
    """2-D Taylor-Green vortex velocity functions ``u(x,y), v(x,y)``.

    A standard, analytically-known incompressible-flow initial condition:
    ``u = sin(x)cos(y)``, ``v = -cos(x)sin(y)`` on a ``[0, 2*pi]^2`` periodic
    domain. It is exactly divergence-free and, under the linearised Stokes
    (low-Re) approximation, its kinetic energy decays monotonically as
    ``exp(-4*viscosity*t)`` — a convenient, well-known sanity check for a
    Navier-Stokes time stepper.
    """
    import jax.numpy as jnp

    def u_fn(x, y):
        return jnp.sin(x) * jnp.cos(y)

    def v_fn(x, y):
        return -jnp.cos(x) * jnp.sin(y)

    return u_fn, v_fn


def shear_layer_velocity_fns(sigma: float = 15.0, width: float = 0.05) -> Tuple[Callable, Callable]:
    """2-D double shear-layer velocity functions ``u(x,y), v(x,y)``.

    Another standard incompressible-flow benchmark IC (e.g. Bell, Colella &
    Glaz 1989 / Minion & Brown 1997): two counter-shearing horizontal
    layers with a small sinusoidal perturbation in ``v`` that seeds
    Kelvin-Helmholtz roll-up. Domain assumed ``[0, 2*pi]^2``.
    """
    import jax.numpy as jnp

    def u_fn(x, y):
        return jnp.where(y <= jnp.pi, jnp.tanh(sigma * (y - jnp.pi / 2)),
                          jnp.tanh(sigma * (3 * jnp.pi / 2 - y)))

    def v_fn(x, y):
        return width * jnp.sin(x)

    return u_fn, v_fn


_IC_FACTORY: Dict[str, Callable[[], Tuple[Callable, Callable]]] = {
    "taylor_green": taylor_green_velocity_fns,
    "shear_layer": shear_layer_velocity_fns,
}


# ===========================================================================
# JAXCFDSolver
# ===========================================================================

@SolverRegistry.register(
    name="jax_cfd",
    family="pde",
    description="Real 2-D incompressible Navier-Stokes via Google's differentiable jax-cfd library "
                "(finite-volume, MAC-grid, semi-implicit time stepping, spectral pressure projection).",
    tags=["fluids", "navier_stokes", "cfd", "jax"],
)
class JAXCFDSolver(SolverBase):
    """2-D incompressible Navier-Stokes solver backed by ``jax_cfd.base``.

    This is a thin adapter: it builds a ``jax_cfd.base.grids.Grid``, an
    initial divergence-free velocity field, and a
    ``jax_cfd.base.equations.semi_implicit_navier_stokes`` step function,
    then advances the real jax-cfd simulation for ``steps`` iterations.
    Results are converted from JAX arrays to CPU ``torch.Tensor``s at the
    ``SolverOutput`` boundary (a non-differentiable boundary, matching
    ``JAXBackend.jax_to_torch``'s convention) — the simulation itself runs
    entirely in JAX/jax_cfd.

    Parameters
    ----------
    nx, ny         : grid resolution.
    viscosity      : kinematic viscosity ``nu`` (lower = higher effective Re).
    density        : fluid density (default 1.0).
    domain         : ((x0,x1), (y0,y1)) physical domain; default ``[0, 2*pi]^2``
                     (the natural periodic domain for the built-in ICs).
    max_velocity   : characteristic velocity scale used only to pick a
                     CFL-stable ``dt`` via ``jax_cfd.base.equations.stable_time_step``.
    cfl_safety     : Courant-number safety factor for the stable-timestep estimate.
    initial_condition : "taylor_green" (default), "shear_layer", or a
                     ``(u_fn, v_fn)`` callable pair of the form used by
                     ``jax_cfd.base.initial_conditions.initial_velocity_field``.

    Notes
    -----
    ``jax_cfd`` and ``jax`` are optional dependencies (``pip install jax
    jaxlib jax-cfd``); importing this module never requires them, but
    constructing :class:`JAXCFDSolver` does (raises a clear ``ImportError``
    otherwise).
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        viscosity: float = 1e-2,
        density: float = 1.0,
        domain: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        max_velocity: float = 1.0,
        cfl_safety: float = 0.5,
        initial_condition: Any = "taylor_green",
        peak_wavenumber: float = 3.0,
        seed: int = 0,
    ):
        super().__init__()
        _require_jax_cfd()

        import math

        self.nx = int(nx)
        self.ny = int(ny)
        self.viscosity = float(viscosity)
        self.density = float(density)
        self.domain = domain or ((0.0, 2.0 * math.pi), (0.0, 2.0 * math.pi))
        self.max_velocity = float(max_velocity)
        self.cfl_safety = float(cfl_safety)
        self.initial_condition = initial_condition
        self.peak_wavenumber = float(peak_wavenumber)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    @classmethod
    def from_problem_spec(cls, spec) -> "JAXCFDSolver":
        """Build from a ProblemSpec (pde.kind ~ 'navier_stokes_incompressible').

        Looks up grid/solver params from both ``spec.pde.params`` and
        ``spec.solver_spec['params']`` (the latter is where presets such as
        ``ns_incompressible_2d`` put ``nx``/``ny``/``Re``/``dt`` — see
        ``pinneapple_physics/pde_environment/presets/cfd.py``), mirroring
        ``LBMSolver.from_problem_spec``'s pattern of reading directly off
        the spec rather than requiring a bespoke config object.
        """
        pde_params = dict(spec.pde.params) if hasattr(spec.pde, "params") else {}
        solver_params = dict(spec.solver_spec.get("params", {})) if getattr(spec, "solver_spec", None) else {}
        # solver_spec params (preset-specific numerics) take priority over
        # generic pde params for grid/discretisation settings, but Re read
        # from pde.params is the physically-authoritative one if both exist.
        p = {**pde_params, **solver_params}

        nx = int(p.get("nx", 64))
        ny = int(p.get("ny", 64))
        Umax = float(p.get("Umax", p.get("max_velocity", 1.0)))
        Re = p.get("Re", None)
        if "viscosity" in p:
            viscosity = float(p["viscosity"])
        elif "nu" in p:
            viscosity = float(p["nu"])
        elif Re is not None:
            # Characteristic length taken as the (unit) domain extent.
            viscosity = float(Umax) / float(Re)
        else:
            viscosity = 1e-2

        return cls(
            nx=nx,
            ny=ny,
            viscosity=viscosity,
            density=float(p.get("density", 1.0)),
            max_velocity=Umax,
        )

    # ------------------------------------------------------------------
    def solve_from_spec(self, spec, steps: int = 200, save_every: int = 20) -> SolverOutput:
        """Run from a ProblemSpec and return SolverOutput with flow fields."""
        p = dict(spec.pde.params) if hasattr(spec.pde, "params") else {}
        sp = dict(spec.solver_spec.get("params", {})) if getattr(spec, "solver_spec", None) else {}
        return self.forward(
            steps=int(sp.get("steps", p.get("steps", steps))),
            save_every=int(sp.get("save_every", p.get("save_every", save_every))),
        )

    # ------------------------------------------------------------------
    def _build_initial_velocity(self, grid):
        """Return a jax_cfd divergence-free initial velocity field on *grid*."""
        import jax
        import jax_cfd.base as cfd

        ic = self.initial_condition
        if isinstance(ic, str):
            if ic == "random":
                key = jax.random.PRNGKey(self.seed)
                return cfd.initial_conditions.filtered_velocity_field(
                    key, grid, maximum_velocity=self.max_velocity,
                    peak_wavenumber=self.peak_wavenumber,
                )
            if ic not in _IC_FACTORY:
                raise ValueError(
                    f"Unknown initial_condition '{ic}'. Available: "
                    f"{sorted(_IC_FACTORY.keys())} or 'random', or pass a (u_fn, v_fn) pair."
                )
            u_fn, v_fn = _IC_FACTORY[ic]()
        else:
            u_fn, v_fn = ic  # assume caller passed a (u_fn, v_fn) pair

        return cfd.initial_conditions.initial_velocity_field((u_fn, v_fn), grid, iterations=3)

    # ------------------------------------------------------------------
    def forward(
        self,
        v0: Optional[Any] = None,
        *,
        steps: int = 200,
        save_every: int = 20,
    ) -> SolverOutput:
        """Run the jax-cfd simulation for `steps` semi-implicit NS steps.

        Parameters
        ----------
        v0         : optional pre-built jax_cfd velocity-field tuple
                     (``GridVariable``s); if ``None``, built from
                     ``self.initial_condition``.
        steps      : total timesteps.
        save_every : save macroscopic (u, v) snapshots every N steps.

        Returns
        -------
        SolverOutput
          result           : stacked final (u, v) as a torch.Tensor, shape (2, nx, ny)
          extras['u']      : final x-velocity (nx, ny) torch.Tensor
          extras['v']      : final y-velocity (nx, ny) torch.Tensor
          extras['vel_mag']: final speed field (nx, ny) torch.Tensor
          extras['trajectory_u'/'trajectory_v'] : list of saved snapshots
          extras['dt'], extras['kinetic_energy_history']
        """
        _require_jax_cfd()

        import jax.numpy as jnp
        import jax_cfd.base as cfd
        import numpy as np

        grid = cfd.grids.Grid((self.nx, self.ny), domain=self.domain)
        dt = float(cfd.equations.stable_time_step(
            self.max_velocity, self.cfl_safety, self.viscosity, grid,
        ))

        state = v0 if v0 is not None else self._build_initial_velocity(grid)

        step_fn = cfd.equations.semi_implicit_navier_stokes(
            density=self.density,
            viscosity=self.viscosity,
            dt=dt,
            grid=grid,
        )

        def _ke(s) -> float:
            return float(sum(jnp.sum(c.data ** 2) for c in s) / (self.nx * self.ny))

        traj_u, traj_v, ke_history = [], [], [_ke(state)]

        for step in range(steps):
            state = step_fn(state)
            if (step + 1) % max(save_every, 1) == 0:
                traj_u.append(torch.from_numpy(np.array(state[0].data)).clone())
                traj_v.append(torch.from_numpy(np.array(state[1].data)).clone())
                ke_history.append(_ke(state))

        u_final = torch.from_numpy(np.array(state[0].data)).clone()
        v_final = torch.from_numpy(np.array(state[1].data)).clone()
        result = torch.stack([u_final, v_final], dim=0)

        return SolverOutput(
            result=result,
            losses={},
            extras={
                "u": u_final,
                "v": v_final,
                "vel_mag": torch.sqrt(u_final ** 2 + v_final ** 2),
                "trajectory_u": traj_u,
                "trajectory_v": traj_v,
                "dt": dt,
                "viscosity": self.viscosity,
                "density": self.density,
                "kinetic_energy_history": ke_history,
                "grid_shape": (self.nx, self.ny),
                "domain": self.domain,
            },
        )
