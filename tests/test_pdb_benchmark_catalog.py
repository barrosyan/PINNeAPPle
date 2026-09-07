"""Tests for ``pinneapple_pdb``'s small, curated, named-benchmark catalog
(``pinneapple_pdb/benchmarks.py``), added to close the gap
``ROADMAP_PHYSICS_AI_HUB.md`` section P3.2 and
``pinneapple_llm.guardrail.PhysicsGuardrail._load_reference_from_upd_zarr``'s
docstring both flagged: until this pass, ``pinneapple_pdb`` had no way to
resolve a friendly name (e.g. ``"lane_emden_n1.5"``) to a real reference
dataset -- only a file-path fetch existed.

Both catalog entries here are the astrophysically standard Lane-Emden
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


def test_list_benchmarks_contains_lane_emden_entries():
    names = list_benchmarks()
    assert "lane_emden_n1.5" in names
    assert "lane_emden_n3" in names
    # Small, curated -- not a comprehensive suite (see benchmarks.py's module docstring).
    assert len(names) == 2


def test_benchmark_catalog_returns_all_entries_by_name():
    catalog = benchmark_catalog()
    assert set(catalog.keys()) == {"lane_emden_n1.5", "lane_emden_n3"}
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
