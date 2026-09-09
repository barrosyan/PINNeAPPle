"""Tests for pinneapple_llm.catalog_match.resolve_problem."""
from __future__ import annotations

import os

from pinneapple_llm.catalog_match import resolve_problem, _keyword_shortlist, _tokenize
from pinneapple_physics.pde_environment.presets.registry import list_presets


def test_tokenize_strips_stopwords_and_short_tokens():
    tokens = _tokenize("I need to solve the Burgers equation in 1D")
    assert "burgers" in tokens
    assert "equation" in tokens
    assert "the" not in tokens and "in" not in tokens


def test_keyword_shortlist_prioritizes_overlap():
    catalog = [
        {"name": "burgers_1d", "accepted_kwargs": ["nu"]},
        {"name": "lane_emden", "accepted_kwargs": ["n"]},
        {"name": "heat_conduction_1d", "accepted_kwargs": ["k"]},
    ]
    shortlist = _keyword_shortlist("I want to simulate burgers turbulence", catalog, top_k=2)
    names = [c["name"] for c in shortlist]
    assert "burgers_1d" in names


def test_resolve_problem_falls_back_without_llm_credentials():
    # No API key is configured in this environment -- resolve_problem must
    # never raise, and must fall back to the deterministic keyword match.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(var, None)

    result = resolve_problem("burgers equation turbulence one dimensional")
    assert result.used_llm is False
    assert result.preset is not None
    assert result.preset in list_presets()
    assert len(result.shortlist) > 0


def test_resolve_problem_shortlist_matches_real_catalog():
    result = resolve_problem("heat conduction steady state")
    presets = set(list_presets())
    assert all(name in presets for name in result.shortlist)
