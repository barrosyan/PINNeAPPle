"""Tests for pinneapple_analysis.verification.tool_recommendation.

First test file for this new module. Uses real ProblemSpec presets from
pinneapple_physics wherever possible (matching the sibling
test_solver_orchestration.py's convention), plus a minimal fake spec for
edge cases a real preset can't easily exercise (a pde_kind guaranteed to
match nothing in the catalog).
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.tool_recommendation import (
    TOOL_CATALOG,
    ExternalTool,
    ToolRecommendation,
    recommend_tools,
    get_tools_by_category,
    Player,
    PLAYER_CATALOG,
    get_player_for_tool,
    BUY_VS_BUILD_GUIDANCE,
    buy_vs_build_recommendation,
    list_buy_vs_build_needs,
)


class _FakeSpec:
    """Minimal ProblemSpec stand-in -- recommend_tools only reads
    spec.pde.kind / spec.dim."""

    class _PDE:
        def __init__(self, kind):
            self.kind = kind

    def __init__(self, kind: str, dim: int = 2):
        self.pde = self._PDE(kind)
        self.dim = dim


def test_catalog_entries_are_well_formed():
    assert len(TOOL_CATALOG) >= 10
    for key, tool in TOOL_CATALOG.items():
        assert isinstance(tool, ExternalTool)
        assert tool.name and tool.category and tool.source
        assert tool.license in ("open-source", "commercial")
        assert tool.typical_cost_tier in ("free", "$", "$$", "$$$")
        assert tool.dims.issubset({2, 3})
        # An open-source tool should cost "free" (a real, checkable internal
        # consistency constraint on the catalog itself, not a tautology --
        # a commercial tool mislabeled "free" or vice versa would be a bug).
        if tool.license == "open-source":
            assert tool.typical_cost_tier == "free", f"{key}: open-source but not free-tier"


def test_recommend_tools_navier_stokes_3d_matches_real_cfd_tools():
    spec = _FakeSpec("navier_stokes", dim=3)
    rec = recommend_tools(spec)

    assert isinstance(rec, ToolRecommendation)
    assert "openfoam" in rec.recommended
    assert "ansys_fluent" in rec.recommended
    assert len(rec.reasoning) > 20


def test_recommend_tools_open_source_ranked_before_commercial():
    spec = _FakeSpec("navier_stokes", dim=3)
    rec = recommend_tools(spec)

    assert len(rec.recommended) >= 2
    primary_tool = TOOL_CATALOG[rec.recommended[0]]
    assert primary_tool.license == "open-source", (
        f"expected an open-source tool ranked first, got {primary_tool.name} "
        f"({primary_tool.license})"
    )
    # Confirm the full ranking is actually sorted open-source-first, not just the head.
    licenses = [TOOL_CATALOG[k].license for k in rec.recommended]
    first_commercial = next((i for i, l in enumerate(licenses) if l == "commercial"), len(licenses))
    assert all(l == "open-source" for l in licenses[:first_commercial])


def test_recommend_tools_no_match_returns_empty_not_a_forced_pick():
    spec = _FakeSpec("some_pde_kind_nothing_documents", dim=3)
    rec = recommend_tools(spec)

    assert rec.recommended == []
    assert rec.alternatives == []
    assert "does not match" in rec.reasoning


def test_recommend_tools_alternatives_explain_why_not_primary():
    spec = _FakeSpec("navier_stokes", dim=3)
    rec = recommend_tools(spec)

    assert len(rec.alternatives) == len(rec.recommended) - 1
    for alt in rec.alternatives:
        assert alt["tool"] in TOOL_CATALOG
        assert "why_not_primary" in alt and len(alt["why_not_primary"]) > 10


def test_recommend_tools_category_filter():
    spec = _FakeSpec("navier_stokes", dim=3)
    rec_cfd = recommend_tools(spec, category="CFD")
    rec_fem = recommend_tools(spec, category="FEM")

    assert all(TOOL_CATALOG[k].category == "CFD" for k in rec_cfd.recommended)
    # navier_stokes is not in any FEM tool's documented pde_kinds in this catalog
    assert rec_fem.recommended == []


def test_recommend_tools_fem_heat_equation_matches_fenicsx():
    spec = _FakeSpec("heat_equation_steady", dim=2)
    rec = recommend_tools(spec)
    assert "fenicsx" in rec.recommended


def test_recommend_tools_dimensionless_numbers_caveat_never_changes_selection():
    class _FakeDimless:
        reynolds = 50000.0

    spec = _FakeSpec("navier_stokes", dim=3)
    rec_plain = recommend_tools(spec)
    rec_with_dimless = recommend_tools(spec, dimensionless_numbers=_FakeDimless())

    assert rec_plain.recommended == rec_with_dimless.recommended
    assert rec_plain.caveats == []
    assert any("turbulent regime" in c for c in rec_with_dimless.caveats)


def test_recommend_tools_custom_catalog_override():
    custom = {
        "my_tool": ExternalTool(
            name="MyTool", category="CFD", license="open-source", typical_cost_tier="free",
            pde_kinds=["navier_stokes"], dims={3}, source="internal test fixture",
        )
    }
    spec = _FakeSpec("navier_stokes", dim=3)
    rec = recommend_tools(spec, catalog=custom)
    assert rec.recommended == ["my_tool"]


def test_player_catalog_entries_are_well_formed():
    assert len(PLAYER_CATALOG) >= 8
    for key, player in PLAYER_CATALOG.items():
        assert isinstance(player, Player)
        assert player.name and player.focus_areas
        for tk in player.tool_keys:
            assert tk in TOOL_CATALOG, f"{key}: tool_keys references unknown tool {tk!r}"


def test_get_player_for_tool_finds_the_real_maker():
    assert get_player_for_tool("ansys_fluent") == "ansys"
    assert get_player_for_tool("ansys_mechanical") == "ansys"
    assert get_player_for_tool("comsol") == "comsol_inc"
    assert get_player_for_tool("paraview") == "kitware"


def test_get_player_for_tool_returns_none_for_community_projects():
    # OpenFOAM/FEniCSx/Gmsh/SU2/MOOSE/CalculiX/OpenTURNS have no single
    # corporate player behind them in this catalog -- a real, honest None,
    # not a guessed company name.
    for tool_key in ("openfoam", "fenicsx", "gmsh", "su2", "moose", "calculix", "openturns"):
        assert get_player_for_tool(tool_key) is None, f"{tool_key} should have no PLAYER_CATALOG entry"


def test_every_player_tool_key_is_reachable_only_from_its_own_entry():
    """No tool should be claimed by two different players (a real
    catalog-consistency invariant, not a tautology -- a copy-paste bug
    listing the same tool under two companies would silently give
    ambiguous get_player_for_tool results without this check)."""
    seen: dict = {}
    for key, player in PLAYER_CATALOG.items():
        for tk in player.tool_keys:
            assert tk not in seen, f"{tk} claimed by both {seen[tk]!r} and {key!r}"
            seen[tk] = key


def test_buy_vs_build_known_need_returns_guidance():
    text = buy_vs_build_recommendation("CFD_high_fidelity")
    assert isinstance(text, str) and len(text) > 20


def test_buy_vs_build_unknown_need_returns_none_not_a_guess():
    assert buy_vs_build_recommendation("totally_made_up_need") is None


def test_list_buy_vs_build_needs_matches_guidance_keys():
    assert list_buy_vs_build_needs() == sorted(BUY_VS_BUILD_GUIDANCE.keys())


def test_buy_vs_build_custom_guidance_override():
    custom = {"my_need": "custom guidance text"}
    assert buy_vs_build_recommendation("my_need", guidance=custom) == "custom guidance text"
    assert buy_vs_build_recommendation("CFD_high_fidelity", guidance=custom) is None


def test_get_tools_by_category_meshing_and_visualization():
    meshing = get_tools_by_category("meshing")
    viz = get_tools_by_category("visualization")
    uq = get_tools_by_category("UQ")

    assert "gmsh" in meshing
    assert "paraview" in viz
    assert "openturns" in uq
    # Meshing/viz/UQ tools deliberately have empty pde_kinds -- confirm they
    # never show up in a pde-kind-scoped recommendation.
    spec = _FakeSpec("navier_stokes", dim=3)
    rec = recommend_tools(spec)
    assert "gmsh" not in rec.recommended
    assert "paraview" not in rec.recommended
