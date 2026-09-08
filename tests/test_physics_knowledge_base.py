"""Tests for the phenomenon -> equation -> parameters -> assumptions knowledge base."""
from __future__ import annotations

import importlib

import pytest

from pinneapple_problemdesign.knowledge.physics_knowledge import (
    PHENOMENON_KNOWLEDGE_BASE,
    PhenomenonEntry,
    lookup_phenomenon,
)


def test_knowledge_base_has_representative_coverage():
    # Aim for 15-25 real entries per the task's representative-subset target.
    assert 15 <= len(PHENOMENON_KNOWLEDGE_BASE) <= 40
    for entry in PHENOMENON_KNOWLEDGE_BASE:
        assert isinstance(entry, PhenomenonEntry)
        assert entry.phenomenon
        assert entry.governing_equation
        assert entry.typical_parameters
        assert entry.assumptions
        assert entry.preset_module
        assert entry.preset_function


def test_knowledge_base_spans_required_domains():
    modules = {e.preset_module for e in PHENOMENON_KNOWLEDGE_BASE}
    required = {
        "pinneapple_physics.pde_environment.presets.cfd",
        "pinneapple_physics.pde_environment.presets.engineering",
        "pinneapple_physics.pde_environment.presets.solid_mechanics",
    }
    assert required.issubset(modules)
    # at least one more domain beyond the three required ones
    assert len(modules) >= 4


@pytest.mark.parametrize(
    "keyword",
    ["navier_stokes", "poiseuille", "lame", "kepler", "lane_emden", "shock tube"],
)
def test_lookup_phenomenon_returns_matches_for_known_keywords(keyword):
    hits = lookup_phenomenon(keyword)
    assert len(hits) >= 1
    for hit in hits:
        haystack = " ".join(
            [hit.phenomenon, hit.governing_equation, hit.preset_function]
        ).lower()
        assert keyword.lower() in haystack


def test_lookup_phenomenon_is_case_insensitive_and_handles_misses():
    assert lookup_phenomenon("NAVIER_STOKES") == lookup_phenomenon("navier_stokes")
    assert lookup_phenomenon("this_keyword_should_not_exist_anywhere") == []
    assert lookup_phenomenon("") == []


def test_every_entry_preset_module_is_real_and_importable():
    """Every preset_module must be a real, importable module, and
    preset_function must actually be defined (or registered) in it --
    this is the traceability guarantee the knowledge base promises."""
    for entry in PHENOMENON_KNOWLEDGE_BASE:
        mod = importlib.import_module(entry.preset_module)
        assert hasattr(mod, entry.preset_function), (
            f"{entry.preset_module} has no attribute {entry.preset_function!r} "
            f"(entry: {entry.phenomenon!r})"
        )
        fn = getattr(mod, entry.preset_function)
        assert callable(fn)


def test_sourced_presets_actually_build_a_problem_spec():
    """Spot-check a handful of entries: calling the real preset function
    with defaults must produce a working ProblemSpec, confirming the
    knowledge-base entry is not just plausible-looking but tied to a
    genuinely functioning preset."""
    from pinneapple_physics.pde_environment.spec import ProblemSpec

    sample_names = {
        "ns_incompressible_2d_default",
        "kepler_two_body_orbit",
        "thick_walled_cylinder_lame_default",
    }
    for entry in PHENOMENON_KNOWLEDGE_BASE:
        if entry.preset_function in sample_names:
            mod = importlib.import_module(entry.preset_module)
            fn = getattr(mod, entry.preset_function)
            spec = fn()
            assert isinstance(spec, ProblemSpec)
