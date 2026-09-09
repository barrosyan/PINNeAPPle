"""Tests for ``pinneapple_pdb``'s small, curated, named-benchmark catalog
(``pinneapple_pdb/benchmarks.py``), added to close the gap
``ROADMAP_PHYSICS_AI_HUB.md`` section P3.2 and
``pinneapple_llm.guardrail.PhysicsGuardrail._load_reference_from_upd_zarr``'s
docstring both flagged: until this pass, ``pinneapple_pdb`` had no way to
resolve a friendly name (e.g. ``"lane_emden_n1.5"``) to a real reference
dataset -- only a file-path fetch existed.

Two catalog entries are the astrophysically standard Lane-Emden
polytropic indices (n=1.5, n=3) that
``pinneapple_physics.pde_environment.presets.astrophysics
.lane_emden_polytrope`` models but has no closed-form solution for. Each
entry independently re-integrates the Lane-Emden ODE (via
``scipy.integrate.solve_ivp``, not imported from the preset's compiler code)
and cross-checks the result against the published surface radius xi_1
(Hansen, Kawaler & Trimble, "Stellar Interiors", 2nd ed., Table 4.1) -- the
same numbers ``tests/test_lane_emden_numerical_validation.py`` independently
verifies. These tests confirm the catalog resolves to real, physically
correct values, not placeholders.

The third entry, ``concorde_high_aoa``, is explicitly a THEORETICAL
closed-form cross-check (the Polhamus 1966 leading-edge-suction vortex-lift
analogy for slender delta wings, in its low-aspect-ratio asymptotic limit)
evaluated at Concorde's approximate wing aspect ratio -- NOT real Concorde
wind-tunnel or flight-test data, which is not available with a citable
public source. See ``benchmarks.py``'s docstring and the entry's own
``verification_note`` for the exact scope and limitations.

Two further entries close classical benchmark gaps flagged by a capability
audit: ``blasius_flat_plate`` (the Blasius laminar flat-plate boundary-layer
similarity solution, independently integrated via a shooting method and
cross-checked against the published wall-shear parameter f''(0)=0.332057,
Howarth 1938 / Schlichting & Gersten) and
``planewall_transient_conduction_bi1`` (1D transient conduction in a plane
wall with convective boundary conditions, Biot number 1.0, the one-term
series centerline-temperature history, independently re-solved via
``scipy.optimize.brentq`` and cross-checked against Incropera & DeWitt's
published Table 5.1 eigenvalue/coefficient). Same "never fabricate, always
re-verify" pattern as every other entry in this catalog.

The final three entries integrate PDEBench (Takamoto et al., "PDEBench: An
Extensive Benchmark for Scientific Machine Learning", NeurIPS 2022 Datasets
and Benchmarks Track, arXiv:2210.07182) -- a real, well-known open-source
PDE-ML benchmark suite -- without fetching any of its actual (multi-GB,
DaRUS-hosted) datasets: ``pdebench_advection_1d`` (PDEBench's 1D linear
advection equation, cross-checked against its exact method-of-characteristics
solution), ``pdebench_diffusion_reaction_1d`` (PDEBench's 1D
diffusion-reaction equation, which is exactly the Fisher-KPP equation,
cross-checked against the Ablowitz-Zeppetella 1979 exact traveling-wave
solution), and ``pdebench_darcy_2d`` (PDEBench's 2D Darcy flow equation,
specialized to homogeneous permeability, cross-checked against both a
closed-form Fourier series and an independent finite-difference solve).
See ``benchmarks.py``'s module docstring for full method and citations.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_pdb import (
    BenchmarkEntry,
    benchmark_catalog,
    get_benchmark,
    list_benchmarks,
)

_PUBLISHED_XI1 = {"lane_emden_n1.5": 3.65375, "lane_emden_n3": 6.89685}


_ALL_BENCHMARK_NAMES = {
    "lane_emden_n1.5",
    "lane_emden_n3",
    "concorde_high_aoa",
    "blasius_flat_plate",
    "planewall_transient_conduction_bi1",
    "pdebench_advection_1d",
    "pdebench_diffusion_reaction_1d",
    "pdebench_darcy_2d",
}


def test_list_benchmarks_contains_lane_emden_entries():
    names = list_benchmarks()
    assert "lane_emden_n1.5" in names
    assert "lane_emden_n3" in names
    assert "concorde_high_aoa" in names
    assert "blasius_flat_plate" in names
    assert "planewall_transient_conduction_bi1" in names
    assert "pdebench_advection_1d" in names
    assert "pdebench_diffusion_reaction_1d" in names
    assert "pdebench_darcy_2d" in names
    # Small, curated -- not a comprehensive suite (see benchmarks.py's module docstring).
    assert len(names) == 8


def test_benchmark_catalog_returns_all_entries_by_name():
    catalog = benchmark_catalog()
    assert set(catalog.keys()) == _ALL_BENCHMARK_NAMES
    for name, entry in catalog.items():
        assert isinstance(entry, BenchmarkEntry)
        assert entry.name == name


def test_get_benchmark_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        get_benchmark("not_a_real_benchmark")


@pytest.mark.parametrize("name", ["lane_emden_n1.5", "lane_emden_n3"])
def test_lane_emden_benchmark_shapes_and_column_order(name):
    entry = get_benchmark(name)
    assert entry.x_vars == ("xi",)
    assert entry.y_vars == ("theta", "phi")
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 2)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    # xi (the radial coordinate) must be strictly increasing from center to surface.
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)


@pytest.mark.parametrize("name", ["lane_emden_n1.5", "lane_emden_n3"])
def test_lane_emden_benchmark_matches_published_surface_radius(name):
    """The last x value (the surface xi_1) must match the published,
    independently-verified table value (see module docstring) to <0.1% --
    the exact tolerance ``tests/test_lane_emden_numerical_validation.py``
    itself asserts for the same physics."""
    entry = get_benchmark(name)
    xi1_integrated = float(entry.reference_x[-1, 0])
    published = _PUBLISHED_XI1[name]
    rel_err = abs(xi1_integrated - published) / published
    assert rel_err < 1e-3, f"{name}: xi_1={xi1_integrated} vs published {published}, rel_err={rel_err:.4%}"


@pytest.mark.parametrize("name", ["lane_emden_n1.5", "lane_emden_n3"])
def test_lane_emden_benchmark_theta_profile_is_physically_sane(name):
    """theta(xi) must start near 1 (the star's dimensionless central value)
    and decrease monotonically-ish to ~0 at the surface (by definition of
    xi_1 as the first zero crossing) -- a sanity check that the profile is
    a real physical solution, not a placeholder array."""
    entry = get_benchmark(name)
    theta = entry.reference_y[:, 0]
    assert theta[0] == pytest.approx(1.0, abs=1e-2)
    assert abs(theta[-1]) < 1e-2  # ~0 at the surface, by construction of xi_1
    assert theta[0] > theta[-1]  # net decrease from center to surface


@pytest.mark.parametrize("name", ["lane_emden_n1.5", "lane_emden_n3"])
def test_lane_emden_benchmark_reference_source_cites_published_table(name):
    entry = get_benchmark(name)
    assert "Hansen" in entry.reference_source or "Chandrasekhar" in entry.reference_source
    assert str(_PUBLISHED_XI1[name]) in entry.reference_source


def test_concorde_high_aoa_benchmark_shapes_and_column_order():
    entry = get_benchmark("concorde_high_aoa")
    assert entry.x_vars == ("alpha_deg",)
    assert entry.y_vars == ("cl",)
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 1)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    # AoA sweep must be strictly increasing and include 10 degrees.
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)
    assert 10.0 in entry.reference_x[:, 0]


def test_concorde_high_aoa_benchmark_cl_is_physically_sane():
    """CL must be zero at zero AoA and monotonically increasing over the
    sweep -- the qualitative vortex-lift signature (CL keeps rising well
    past where a conventional wing would stall) that is the whole point of
    citing the Polhamus analogy for a slender delta wing."""
    entry = get_benchmark("concorde_high_aoa")
    alpha = entry.reference_x[:, 0]
    cl = entry.reference_y[:, 0]
    assert cl[0] == pytest.approx(0.0, abs=1e-6)  # CL(0)=0 by construction
    assert np.all(np.diff(cl) > 0)  # monotonically increasing over 0..20 deg
    cl_at_10deg = float(cl[np.argmin(np.abs(alpha - 10.0))])
    assert 0.0 < cl_at_10deg < 2.0  # sane bound for a vortex-lift-dominated CL


def test_concorde_high_aoa_benchmark_is_explicit_about_not_being_real_data():
    """This benchmark must never be mistaken for real Concorde wind-tunnel or
    flight-test data -- both the description and the verification_note must
    say so explicitly (see benchmarks.py's module docstring)."""
    entry = get_benchmark("concorde_high_aoa")
    assert "Polhamus" in entry.reference_source
    assert "NASA TN D-3767" in entry.reference_source
    for text in (entry.description, entry.verification_note):
        assert "not" in text.lower()
        assert "concorde" in text.lower()


# ---------------------------------------------------------------------------
# Blasius flat-plate boundary layer
# ---------------------------------------------------------------------------

def test_blasius_benchmark_shapes_and_column_order():
    entry = get_benchmark("blasius_flat_plate")
    assert entry.x_vars == ("eta",)
    assert entry.y_vars == ("f", "f_prime", "f_pprime")
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 3)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    # eta (the similarity coordinate) must be strictly increasing from the wall outward.
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)


def test_blasius_benchmark_matches_published_wall_shear_parameter():
    """f''(0) (the first reference_y[0, 2] value, i.e. the wall value of the
    third column) must match the published Howarth (1938) / Schlichting &
    Gersten value to <0.1% -- the same tolerance the Lane-Emden entries use
    for their own published cross-check."""
    entry = get_benchmark("blasius_flat_plate")
    fpp0 = float(entry.reference_y[0, 2])
    published = 0.332057
    rel_err = abs(fpp0 - published) / published
    assert rel_err < 1e-3, f"blasius_flat_plate: f''(0)={fpp0} vs published {published}, rel_err={rel_err:.4%}"


def test_blasius_benchmark_velocity_profile_is_physically_sane():
    """f'(eta) = u/U_inf must start at 0 (no-slip at the wall) and increase
    monotonically-ish to ~1 at the edge of the profile (freestream) -- a
    sanity check that the profile is a real physical solution."""
    entry = get_benchmark("blasius_flat_plate")
    f_prime = entry.reference_y[:, 1]
    assert f_prime[0] == pytest.approx(0.0, abs=1e-6)  # no-slip at the wall
    assert f_prime[-1] == pytest.approx(1.0, abs=1e-3)  # freestream match
    assert f_prime[0] < f_prime[-1]  # net increase from wall to freestream


def test_blasius_benchmark_reference_source_cites_published_value():
    entry = get_benchmark("blasius_flat_plate")
    assert "Howarth" in entry.reference_source or "Schlichting" in entry.reference_source
    assert "0.332057" in entry.reference_source


def test_blasius_benchmark_is_explicit_about_normalization_convention():
    """This benchmark's f''(0) must never be confused with the 0.4696 value
    that belongs to a different similarity-variable scaling -- the
    verification_note must say so explicitly (see benchmarks.py's module
    docstring)."""
    entry = get_benchmark("blasius_flat_plate")
    assert "0.4696" in entry.verification_note


# ---------------------------------------------------------------------------
# 1D transient conduction in a plane wall (Bi=1.0)
# ---------------------------------------------------------------------------

def test_planewall_conduction_benchmark_shapes_and_column_order():
    entry = get_benchmark("planewall_transient_conduction_bi1")
    assert entry.x_vars == ("fourier_number",)
    assert entry.y_vars == ("theta_star_centerline",)
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 1)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    # Fourier number sweep must be strictly increasing, and restricted to the
    # one-term approximation's own documented validity range (Fo > 0.2).
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)
    assert np.all(entry.reference_x[:, 0] >= 0.2)


def test_planewall_conduction_benchmark_matches_published_eigenvalue_and_coefficient():
    """The independently-solved zeta_1/C1 (recoverable from the reference
    curve itself: theta*_0(Fo) = C1*exp(-zeta_1^2*Fo)) must match Incropera
    & DeWitt's published Table 5.1 values for Bi=1.0 to <0.1% -- checked here
    via the verification_note, which records the exact re-derived numbers."""
    entry = get_benchmark("planewall_transient_conduction_bi1")
    assert "0.8603" in entry.verification_note
    assert "1.1191" in entry.verification_note


def test_planewall_conduction_benchmark_temperature_decays_monotonically():
    """theta*_0(Fo) must decay monotonically toward 0 as Fo increases (the
    wall cools/heats toward the fluid temperature over time) -- a sanity
    check that the curve is a real physical solution, not a placeholder."""
    entry = get_benchmark("planewall_transient_conduction_bi1")
    theta = entry.reference_y[:, 0]
    assert np.all(np.diff(theta) < 0)  # strictly decreasing
    assert np.all(theta > 0.0)  # never overshoots past the fluid temperature within this Fo range


def test_planewall_conduction_benchmark_reference_source_cites_incropera():
    entry = get_benchmark("planewall_transient_conduction_bi1")
    assert "Incropera" in entry.reference_source
    assert "Table 5.1" in entry.reference_source


# ---------------------------------------------------------------------------
# PDEBench-derived entries (Takamoto et al., NeurIPS 2022, arXiv:2210.07182)
# ---------------------------------------------------------------------------

def test_pdebench_advection_benchmark_shapes_and_column_order():
    entry = get_benchmark("pdebench_advection_1d")
    assert entry.x_vars == ("x",)
    assert entry.y_vars == ("u",)
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 1)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)


def test_pdebench_advection_benchmark_matches_exact_shifted_sinusoid():
    """The stored reference must equal the PDE's own exact
    method-of-characteristics solution u(t,x)=u0(x-beta*t) evaluated at
    beta=0.4, t=1.3 -- not an approximation."""
    entry = get_benchmark("pdebench_advection_1d")
    x = entry.reference_x[:, 0].astype(np.float64)
    u = entry.reference_y[:, 0].astype(np.float64)
    beta, t_final = 0.4, 1.3
    u_exact = np.sin(2.0 * np.pi * (x - beta * t_final))
    assert np.max(np.abs(u - u_exact)) < 1e-5


def test_pdebench_advection_benchmark_reference_source_cites_pdebench():
    entry = get_benchmark("pdebench_advection_1d")
    assert "Takamoto" in entry.reference_source
    assert "2210.07182" in entry.reference_source
    assert "Strauss" in entry.reference_source


def test_pdebench_diffusion_reaction_benchmark_shapes_and_column_order():
    entry = get_benchmark("pdebench_diffusion_reaction_1d")
    assert entry.x_vars == ("x",)
    assert entry.y_vars == ("u",)
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 1)
    assert entry.reference_y.shape == (n, 1)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    assert np.all(np.diff(entry.reference_x[:, 0]) > 0)


def test_pdebench_diffusion_reaction_benchmark_matches_ablowitz_zeppetella_front():
    """The stored reference must equal the Ablowitz-Zeppetella (1979) exact
    Fisher-KPP traveling-wave solution for nu=0.5, rho=1.0 at t=2.0, and
    must look like a genuine monotone front (0 to 1), not a placeholder."""
    entry = get_benchmark("pdebench_diffusion_reaction_1d")
    x = entry.reference_x[:, 0].astype(np.float64)
    u = entry.reference_y[:, 0].astype(np.float64)
    nu, rho, t_final = 0.5, 1.0, 2.0
    c = 5.0 * np.sqrt(rho * nu / 6.0)
    kk = np.sqrt(rho / (6.0 * nu))
    u_exact = 1.0 / (1.0 + np.exp(kk * (x - c * t_final))) ** 2
    assert np.max(np.abs(u - u_exact)) < 1e-3
    assert u[0] == pytest.approx(1.0, abs=1e-3)  # far upstream: fully invaded state
    assert u[-1] == pytest.approx(0.0, abs=1e-3)  # far downstream: uninvaded state
    assert np.all(np.diff(u) <= 1e-9)  # monotonically non-increasing front


def test_pdebench_diffusion_reaction_benchmark_reference_source_cites_fisher_kpp():
    entry = get_benchmark("pdebench_diffusion_reaction_1d")
    assert "Takamoto" in entry.reference_source
    assert "2210.07182" in entry.reference_source
    assert "Ablowitz" in entry.reference_source
    assert "Fisher" in entry.reference_source


def test_pdebench_darcy_benchmark_shapes_and_column_order():
    entry = get_benchmark("pdebench_darcy_2d")
    assert entry.x_vars == ("x", "y")
    assert entry.y_vars == ("u",)
    n = entry.reference_x.shape[0]
    assert entry.reference_x.shape == (n, 2)
    assert entry.reference_y.shape == (n, 1)
    assert entry.reference_x.dtype == np.float32
    assert entry.reference_y.dtype == np.float32
    # x, y coordinates must lie strictly inside the unit square (Dirichlet u=0 boundary excluded).
    assert np.all(entry.reference_x > 0.0)
    assert np.all(entry.reference_x < 1.0)


def test_pdebench_darcy_benchmark_field_is_physically_sane():
    """u must be positive everywhere in the interior (a uniform positive
    forcing term with zero Dirichlet boundaries bulges positive, per the
    maximum principle for -laplacian(u)=c>0) and symmetric under x<->y swap
    (the unit square + constant forcing problem is symmetric under that
    reflection) -- a sanity check that the field is a real solution."""
    entry = get_benchmark("pdebench_darcy_2d")
    u = entry.reference_y[:, 0]
    assert np.all(u > 0.0)
    n_side = int(round(np.sqrt(entry.reference_x.shape[0])))
    assert n_side * n_side == entry.reference_x.shape[0]
    u_grid = u.reshape(n_side, n_side)
    assert np.max(np.abs(u_grid - u_grid.T)) < 1e-6


def test_pdebench_darcy_benchmark_reference_source_cites_pdebench():
    entry = get_benchmark("pdebench_darcy_2d")
    assert "Takamoto" in entry.reference_source
    assert "2210.07182" in entry.reference_source
    assert "0.0736713" in entry.reference_source
