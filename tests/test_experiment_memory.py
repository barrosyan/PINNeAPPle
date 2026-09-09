"""Tests for pinneapple_registry.experiment_memory.ExperimentMemory.

These exercise the real, temp-dir-backed SQLite ``ExperimentStore`` that
``ExperimentMemory`` wraps -- no mocking of storage. ``query_similar`` is
checked with a genuine, checkable TF-IDF ranking assertion (a lexically
similar past record must outrank an unrelated one for a given query), not
just a "doesn't crash" smoke test.
"""
import os

import pytest

from pinneapple_registry.experiment_memory import ExperimentMemory, ExperimentRecord
from pinneapple_registry.experiment_store import ExperimentStore


@pytest.fixture
def memory(tmp_path):
    db_path = os.path.join(str(tmp_path), "experiments.db")
    return ExperimentMemory(db_path=db_path)


def _record(**overrides):
    defaults = dict(
        problem_description="simulate lift and drag for a NACA 0012 airfoil across an AoA sweep",
        physics_domain="cfd",
        approach_summary="LBM D2Q9 with Smagorinsky LES, Cs=0.17",
        outcome="failure",
        confidence_score=0.42,
        metrics={"rmse": 0.31},
        lessons_learned="AoA sweep above 20deg diverged with LBM Smagorinsky Cs=0.17, needed Cs=0.1",
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_log_experiment_returns_real_stored_id(memory):
    run_id = memory.log_experiment(_record())
    assert isinstance(run_id, str)
    assert run_id.isdigit()


def test_round_trip_through_underlying_experiment_store(tmp_path):
    db_path = os.path.join(str(tmp_path), "experiments.db")
    memory = ExperimentMemory(db_path=db_path)
    record = _record()
    run_id = memory.log_experiment(record)

    # Read back directly through the real ExperimentStore this module wraps
    # (not through ExperimentMemory), to confirm no parallel storage exists.
    store = ExperimentStore(db_path)
    runs = store.list_runs("cfd")
    assert len(runs) == 1
    row = runs[0]
    assert str(row["id"]) == run_id
    assert row["status"] == "failure"
    payload = row["final_metrics"]
    assert payload["problem_description"] == record.problem_description
    assert payload["physics_domain"] == "cfd"
    assert payload["approach_summary"] == record.approach_summary
    assert payload["confidence_score"] == 0.42
    assert payload["lessons_learned"] == record.lessons_learned
    assert payload["metrics"] == {"rmse": 0.31}

    # And also confirm the real per-run numeric metrics table got the
    # metrics via the real log_metrics API.
    logged_metrics = store.run_metrics(run_id=row["id"])
    assert any(m["name"] == "rmse" and m["value"] == pytest.approx(0.31) for m in logged_metrics)


def test_query_similar_ranks_lexically_similar_record_above_unrelated(memory):
    airfoil_record = _record(
        problem_description="simulate lift and drag for a NACA 0012 airfoil across an AoA sweep",
        physics_domain="cfd",
        approach_summary="LBM D2Q9 with Smagorinsky LES",
        outcome="failure",
        lessons_learned="AoA sweep above 20deg diverged, needed lower Cs",
    )
    heat_record = _record(
        problem_description="predict steady-state temperature distribution in a copper heat sink fin array",
        physics_domain="heat_conduction",
        approach_summary="PINN with Fourier feature embedding",
        outcome="success",
        lessons_learned="Fourier features fixed high-frequency gradient near fin base",
    )
    orbital_record = _record(
        problem_description="propagate a two-body Keplerian orbit for a satellite constellation",
        physics_domain="orbital_mechanics",
        approach_summary="RK45 numerical integrator",
        outcome="success",
        lessons_learned="RK45 tolerance 1e-9 needed to avoid secular drift",
    )

    memory.log_experiment(airfoil_record)
    memory.log_experiment(heat_record)
    memory.log_experiment(orbital_record)

    query = "airfoil AoA sweep lift drag simulation with LBM turbulence model"
    results = memory.query_similar(query, top_k=3)

    assert len(results) == 3
    top_record, top_score = results[0]
    assert top_record.problem_description == airfoil_record.problem_description
    assert top_score > 0

    # The airfoil record's score must strictly outrank both unrelated
    # records -- a real, checkable ranking assertion, not just "returns
    # something".
    scores_by_domain = {rec.physics_domain: score for rec, score in results}
    assert scores_by_domain["cfd"] > scores_by_domain["heat_conduction"]
    assert scores_by_domain["cfd"] > scores_by_domain["orbital_mechanics"]


def test_query_similar_domain_boost_can_break_ties(memory):
    # Two records with near-identical (but not literally identical, so
    # cosine similarity to the query is < 1.0 and leaves room for the
    # boost to matter) phrasing, differing mainly by domain tag. A query
    # naming the domain explicitly should be nudged toward the
    # matching-domain record via the domain boost.
    memory.log_experiment(
        _record(
            problem_description="steady-state field simulation on a structured mesh grid",
            physics_domain="heat_conduction",
            lessons_learned="Coarse mesh underresolved boundary layer",
        )
    )
    memory.log_experiment(
        _record(
            problem_description="steady-state field simulation over a structured mesh",
            physics_domain="cfd",
            lessons_learned="Coarse mesh underresolved boundary layer",
        )
    )

    results = memory.query_similar(
        "steady-state field simulation on a structured mesh",
        physics_domain="cfd",
        top_k=2,
    )
    assert results[0][0].physics_domain == "cfd"


def test_query_similar_returns_empty_when_nothing_logged(memory):
    assert memory.query_similar("anything at all") == []


def test_get_lessons_for_domain_aggregates_matching_records_only(memory):
    memory.log_experiment(
        _record(
            physics_domain="cfd",
            lessons_learned="Lesson A: refine mesh near leading edge",
        )
    )
    memory.log_experiment(
        _record(
            problem_description="a different cfd run entirely, vortex shedding behind a cylinder",
            physics_domain="cfd",
            lessons_learned="Lesson B: reduce timestep for vortex shedding stability",
        )
    )
    memory.log_experiment(
        _record(
            problem_description="predict steady-state temperature distribution in a heat sink",
            physics_domain="heat_conduction",
            lessons_learned="Lesson C: unrelated domain, should not appear",
        )
    )
    # A record with no lesson at all should be silently skipped.
    memory.log_experiment(
        _record(
            problem_description="another cfd attempt with no lesson recorded",
            physics_domain="cfd",
            lessons_learned=None,
        )
    )

    lessons = memory.get_lessons_for_domain("cfd")
    assert "Lesson A: refine mesh near leading edge" in lessons
    assert "Lesson B: reduce timestep for vortex shedding stability" in lessons
    assert "Lesson C: unrelated domain, should not appear" not in lessons
    assert len(lessons) == 2


def test_get_lessons_for_domain_case_insensitive(memory):
    memory.log_experiment(
        _record(physics_domain="CFD", lessons_learned="Lesson X")
    )
    assert memory.get_lessons_for_domain("cfd") == ["Lesson X"]


def test_construct_from_existing_experiment_store(tmp_path):
    db_path = os.path.join(str(tmp_path), "shared.db")
    store = ExperimentStore(db_path)
    memory = ExperimentMemory(store=store)
    run_id = memory.log_experiment(_record())
    # Same underlying store instance sees the write immediately.
    runs = store.list_runs("cfd")
    assert len(runs) == 1
    assert str(runs[0]["id"]) == run_id


def test_requires_store_or_db_path():
    with pytest.raises(ValueError):
        ExperimentMemory()


def test_rejects_both_store_and_db_path(tmp_path):
    db_path = os.path.join(str(tmp_path), "x.db")
    store = ExperimentStore(db_path)
    with pytest.raises(ValueError):
        ExperimentMemory(store=store, db_path=db_path)
