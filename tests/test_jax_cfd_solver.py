"""Tests for the JAX-CFD solver adapter (pinneapple_simulation.numerical_solvers.jax_cfd_solver).

Uses the real installed `jax_cfd` (https://github.com/google/jax-cfd) library
when available, running a small/fast Taylor-Green-vortex simulation and
checking a real physical property: viscous kinetic-energy decay. If jax_cfd
is genuinely unavailable in this environment, only the registration /
structural / error-path logic is verified (matching the rest of this
package's optional-dependency test convention, e.g. `wavelet`/`pywt`).
"""
from __future__ import annotations

import math

import pytest
import torch

from pinneapple_simulation.numerical_solvers.jax_cfd_solver import (
    JAXCFDSolver,
    jax_cfd_available,
)
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all

register_all()

HAVE_JAX_CFD = jax_cfd_available()


def test_jax_cfd_registered_in_registry():
    """The solver module registers "jax_cfd" regardless of whether jax_cfd is
    installed (registration is decorator-time, at class-definition, not at
    solver-construction time -- constructing JAXCFDSolver is what requires
    the optional dependency)."""
    names = SolverRegistry.list()
    assert "jax_cfd" in names
    spec = SolverRegistry.spec("jax_cfd")
    assert spec.family == "pde"
    assert "navier_stokes" in spec.tags
    assert "cfd" in spec.tags
    assert "jax" in spec.tags


@pytest.mark.skipif(HAVE_JAX_CFD, reason="jax_cfd IS installed; error-path test only applies when absent")
def test_jax_cfd_solver_raises_clear_error_when_missing():
    with pytest.raises(ImportError, match="jax-cfd"):
        JAXCFDSolver(nx=8, ny=8, viscosity=1e-2)


@pytest.mark.skipif(not HAVE_JAX_CFD, reason="jax_cfd not installed")
class TestJAXCFDSolverReal:
    """Real simulation tests -- only run when jax_cfd + jax are actually installed."""

    def test_construct_directly(self):
        solver = JAXCFDSolver(nx=32, ny=32, viscosity=1e-2)
        assert solver.nx == 32 and solver.ny == 32
        assert isinstance(solver, torch.nn.Module)  # SolverBase is an nn.Module

    def test_taylor_green_vortex_short_run_is_finite_and_evolves(self):
        """Run a few real semi_implicit_navier_stokes steps on a Taylor-Green
        vortex IC (a standard, analytically-known divergence-free 2D flow:
        u=sin(x)cos(y), v=-cos(x)sin(y)) and check the output is sane."""
        solver = JAXCFDSolver(nx=32, ny=32, viscosity=1e-2, initial_condition="taylor_green")
        out = solver.forward(steps=10, save_every=5)

        result = out.result
        assert result.shape == (2, 32, 32)
        assert torch.isfinite(result).all(), "simulation blew up (non-finite values)"

        u, v = out.extras["u"], out.extras["v"]
        assert u.shape == (32, 32)
        assert v.shape == (32, 32)
        assert torch.isfinite(out.extras["vel_mag"]).all()

        # It should actually have evolved from a trivial/zero state.
        assert float(u.abs().max()) > 1e-6
        assert float(v.abs().max()) > 1e-6

        # Trajectory snapshots were saved.
        assert len(out.extras["trajectory_u"]) == 2  # steps=10, save_every=5
        assert len(out.extras["trajectory_v"]) == 2

    def test_taylor_green_kinetic_energy_decays_monotonically(self):
        """Physical check: for the (near-Stokes, low max-velocity) Taylor-Green
        vortex, viscosity should monotonically dissipate kinetic energy over
        a short run (KE ~ exp(-4*nu*t) in the linearised limit) -- i.e. the
        solver should not be spuriously injecting energy or blowing up."""
        solver = JAXCFDSolver(nx=48, ny=48, viscosity=1e-2, initial_condition="taylor_green")
        out = solver.forward(steps=30, save_every=5)

        ke_history = out.extras["kinetic_energy_history"]
        assert len(ke_history) >= 2
        assert all(math.isfinite(ke) for ke in ke_history)

        # Monotonic (non-increasing) kinetic energy: viscous dissipation only,
        # no external forcing in this configuration.
        for earlier, later in zip(ke_history, ke_history[1:]):
            assert later <= earlier + 1e-9, (
                f"kinetic energy increased ({earlier} -> {later}); "
                "expected monotonic viscous decay for unforced Taylor-Green vortex"
            )

        # And it should have actually decayed some non-trivial amount, not
        # been numerically frozen.
        assert ke_history[-1] < ke_history[0]

    def test_shear_layer_ic_runs_and_stays_finite(self):
        """Second standard IC (double shear layer) -- different physical
        regime (Kelvin-Helmholtz roll-up), same finiteness/sanity check."""
        solver = JAXCFDSolver(nx=32, ny=32, viscosity=5e-3, initial_condition="shear_layer")
        out = solver.forward(steps=10, save_every=5)
        assert torch.isfinite(out.result).all()

    def test_from_problem_spec_and_solve_from_spec(self):
        """Build a JAXCFDSolver from the real ns_incompressible_2d preset and
        actually run it -- exercising the from_problem_spec/solve_from_spec
        convention shared with LBMSolver."""
        from pinneapple_physics.pde_environment.presets.registry import get_preset

        spec = get_preset("ns_incompressible_2d", Re=200.0)
        solver = JAXCFDSolver.from_problem_spec(spec)
        assert solver.nx > 0 and solver.ny > 0
        assert solver.viscosity > 0

        out = solver.solve_from_spec(spec, steps=5, save_every=5)
        assert torch.isfinite(out.result).all()

    def test_registry_build_returns_working_solver(self):
        """SolverRegistry.build("jax_cfd", ...) must return a real, usable
        JAXCFDSolver instance -- not just a registered class reference."""
        solver = SolverRegistry.build("jax_cfd", nx=16, ny=16, viscosity=2e-2)
        assert isinstance(solver, JAXCFDSolver)

        out = solver.forward(steps=5, save_every=1)
        assert out.result.shape == (2, 16, 16)
        assert torch.isfinite(out.result).all()
        assert len(out.extras["trajectory_u"]) == 5

    def test_invalid_initial_condition_name_raises(self):
        solver = JAXCFDSolver(nx=8, ny=8, viscosity=1e-2, initial_condition="not_a_real_ic")
        with pytest.raises(ValueError, match="Unknown initial_condition"):
            solver.forward(steps=1, save_every=1)
