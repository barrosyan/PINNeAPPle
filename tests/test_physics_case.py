"""Tests for ``PhysicsCase`` (``pinneapple_data/physics_case.py``), the
minimal bridge between the three disjoint "partial canonical
representations" of a physics problem already in this repo: geometry,
``pinneapple_physics.pde_environment.spec.ProblemSpec``, and
``pinneapple_data`` UPD-family results objects -- plus
``pinneapple_pdb.benchmarks.BenchmarkEntry`` for validation.

Where practical, these tests use REAL (not mocked) instances of the
referenced types: a real ``ProblemSpec`` built by the existing
``lane_emden_polytrope`` preset, a real ``BenchmarkEntry`` resolved via
``pinneapple_pdb.benchmarks.get_benchmark("lane_emden_n1.5")`` (which
independently re-verifies its own numerical integration every time it is
built -- see that module's docstring), and a real
``pinneapple_data.physical_sample.PhysicalSample`` for the UPD-results
bridging case.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_data.physical_sample import PhysicalSample
from pinneapple_data.physics_case import BenchmarkComparison, PhysicsCase
from pinneapple_pdb.benchmarks import get_benchmark
from pinneapple_physics.pde_environment.presets.astrophysics import lane_emden_polytrope


# ---------------------------------------------------------------------------
# Construction / optionality
# ---------------------------------------------------------------------------

def test_all_fields_optional():
    """A case may exist before geometry/physics/results are known."""
    case = PhysicsCase()
    assert case.geometry is None
    assert case.geometry_kind is None
    assert case.physics is None
    assert case.solver is None
    assert case.solver_config is None
    assert case.results is None
    assert case.reference_benchmark is None
    assert case.metadata == {}


def test_construction_with_real_problem_spec():
    spec = lane_emden_polytrope(n=1.5)
    case = PhysicsCase(
        name="lane_emden_n1.5_case",
        physics=spec,
        solver="fdm",
        solver_config={"method": "lane_emden_ivp"},
        reference_benchmark="lane_emden_n1.5",
        geometry_kind="mesh",
        metadata={"description": "toy case bridging geometry+physics+solver+results"},
    )
    # physics is a direct reference to the real ProblemSpec, not a copy
    assert case.physics is spec
    assert case.physics.pde.kind == "lane_emden_polytrope"
    assert case.physics.fields == ("theta", "phi")
    assert case.solver == "fdm"
    assert case.reference_benchmark == "lane_emden_n1.5"


# ---------------------------------------------------------------------------
# validate_against_benchmark: dict-based results (simplest UPD-shaped object)
# ---------------------------------------------------------------------------

def test_validate_against_benchmark_requires_results_and_benchmark():
    case = PhysicsCase()
    with pytest.raises(ValueError):
        case.validate_against_benchmark()

    case = PhysicsCase(results={"xi": np.array([0.0]), "theta": np.array([1.0]), "phi": np.array([0.0])})
    with pytest.raises(ValueError):
        case.validate_against_benchmark()


def test_validate_against_benchmark_exact_grid_dict_results():
    """results sampled on exactly the benchmark's own reference grid should
    reproduce it to numerical precision (interpolation onto the identical
    grid is exact)."""
    entry = get_benchmark("lane_emden_n1.5")
    results = {
        "xi": entry.reference_x[:, 0],
        "theta": entry.reference_y[:, 0],
        "phi": entry.reference_y[:, 1],
    }
    case = PhysicsCase(reference_benchmark="lane_emden_n1.5", results=results)
    comparison = case.validate_against_benchmark()

    assert isinstance(comparison, BenchmarkComparison)
    assert comparison.benchmark == "lane_emden_n1.5"
    assert comparison.x_vars == ("xi",)
    assert comparison.y_vars == ("theta", "phi")
    assert comparison.n_reference_points == entry.reference_x.shape[0]
    assert comparison.n_compared_points == entry.reference_x.shape[0]
    assert comparison.rmse < 1e-6
    assert comparison.relative_l2_error < 1e-6
    assert set(comparison.per_variable_rmse) == {"theta", "phi"}


def test_validate_against_benchmark_interpolates_a_different_grid():
    """results on a coarser/different xi grid than the benchmark's own grid
    should still compare cleanly via 1-D linear interpolation."""
    entry = get_benchmark("lane_emden_n1.5")
    idx = np.arange(0, entry.reference_x.shape[0], 3)  # a different (coarser) grid
    results = {
        "xi": entry.reference_x[idx, 0],
        "theta": entry.reference_y[idx, 0],
        "phi": entry.reference_y[idx, 1],
    }
    case = PhysicsCase(reference_benchmark="lane_emden_n1.5", results=results)
    comparison = case.validate_against_benchmark()

    assert comparison.n_reference_points == entry.reference_x.shape[0]
    assert comparison.n_compared_points == entry.reference_x.shape[0]
    # linear interpolation of a subsampled but otherwise identical profile
    # should be small but not necessarily machine-precision
    assert comparison.relative_l2_error < 1e-2
    assert "interpolated" in comparison.note


def test_validate_against_benchmark_wrong_profile_gives_large_error():
    """A deliberately wrong 'results' profile (constant theta/phi) should
    produce a clearly larger error than the exact-grid case above -- this
    is the check-tells-them-apart sanity test."""
    entry = get_benchmark("lane_emden_n1.5")
    results = {
        "xi": entry.reference_x[:, 0],
        "theta": np.zeros_like(entry.reference_y[:, 0]),
        "phi": np.zeros_like(entry.reference_y[:, 1]),
    }
    case = PhysicsCase(reference_benchmark="lane_emden_n1.5", results=results)
    comparison = case.validate_against_benchmark()
    assert comparison.relative_l2_error > 0.1


def test_validate_against_benchmark_missing_variable_raises_informative_error():
    case = PhysicsCase(
        reference_benchmark="lane_emden_n1.5",
        results={"xi": np.array([0.1]), "theta": np.array([0.9])},  # missing "phi"
    )
    with pytest.raises((KeyError, TypeError)):
        case.validate_against_benchmark()


# ---------------------------------------------------------------------------
# validate_against_benchmark: a real PhysicalSample (the real UPD-family
# results object several format readers in pinneapple_simulation and
# pinneapple_design already produce, via meshio_to_upd/cgns_to_upd/etc.)
# ---------------------------------------------------------------------------

def test_validate_against_benchmark_real_physical_sample_results():
    entry = get_benchmark("lane_emden_n1.5")
    state = {
        "xi": torch.as_tensor(entry.reference_x[:, 0]),
        "theta": torch.as_tensor(entry.reference_y[:, 0]),
        "phi": torch.as_tensor(entry.reference_y[:, 1]),
    }
    sample = PhysicalSample(
        state=state,
        domain={"type": "points"},
        provenance={"source": "toy_lane_emden_run"},
    )
    case = PhysicsCase(
        name="lane_emden_n1.5_from_physical_sample",
        physics=lane_emden_polytrope(n=1.5),
        solver="fdm",
        results=sample,
        reference_benchmark="lane_emden_n1.5",
    )
    comparison = case.validate_against_benchmark()
    assert comparison.rmse < 1e-5
    assert comparison.relative_l2_error < 1e-5


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_round_trips_physics_spec():
    spec = lane_emden_polytrope(n=1.5)
    case = PhysicsCase(
        name="case1",
        physics=spec,
        solver="fdm",
        solver_config={"nx": 10},
        reference_benchmark="lane_emden_n1.5",
        metadata={"description": "round trip test"},
    )
    d = case.to_dict()
    restored = PhysicsCase.from_dict(d)

    assert restored.name == "case1"
    assert restored.solver == "fdm"
    assert restored.solver_config == {"nx": 10}
    assert restored.reference_benchmark == "lane_emden_n1.5"
    assert restored.metadata == {"description": "round trip test"}

    assert restored.physics is not None
    assert restored.physics.pde.kind == spec.pde.kind
    assert restored.physics.coords == spec.coords
    assert restored.physics.fields == spec.fields
    assert restored.physics.domain_bounds == spec.domain_bounds
    assert len(restored.physics.conditions) == len(spec.conditions)

    # In-process, the conditions' value_fn callables survive the round trip
    # (copy.deepcopy treats plain functions as atomic), so the rebuilt spec
    # is actually usable, not just descriptive.
    X = np.array([[1e-3]], dtype=np.float32)
    orig_val = spec.conditions[0].value_fn(X, {})
    restored_val = restored.physics.conditions[0].value_fn(X, {})
    np.testing.assert_allclose(orig_val, restored_val)


def test_to_dict_geometry_and_results_are_descriptive_only():
    """geometry/results are honestly NOT round-tripped -- to_dict() records
    only a lightweight description, and from_dict() hands that description
    back rather than pretending to reconstruct the original object."""
    sample = PhysicalSample(state={"x": np.array([1.0, 2.0])}, domain={"type": "points"})
    case = PhysicsCase(results=sample, geometry_kind="mesh")

    d = case.to_dict()
    assert d["results"]["type"] == "PhysicalSample"
    assert "summary" in d["results"]

    restored = PhysicsCase.from_dict(d)
    assert not isinstance(restored.results, PhysicalSample)
    assert restored.results == d["results"]
    assert restored.geometry_kind == "mesh"


def test_to_dict_with_no_physics_or_results():
    case = PhysicsCase(name="empty_case")
    d = case.to_dict()
    assert d["physics"] is None
    assert d["geometry"] is None
    assert d["results"] is None

    restored = PhysicsCase.from_dict(d)
    assert restored.name == "empty_case"
    assert restored.physics is None
