"""Regression tests for ``pinneapple_design.geometry.gen.primitives``'s
mesh-builder registry, found while wiring up LLM-driven CAD-recipe
generation (see ``pinneapple_llm/cad_draft.py``): the registry's
``build_mesh``/``list_builders`` are the real, checkable catalog that
drafting module presents to an LLM, so a correctness bug here silently
degrades every downstream geometry an LLM (or anyone else) requests.

Real bug found and fixed: ``_boolean_engines_available()`` crashed on
``sorted()`` (mixing ``None`` with engine-name strings, since
``trimesh.boolean.engines_available`` includes a literal ``None``
sentinel) and silently swallowed the ``TypeError`` via a bare
``except Exception``, so it ALWAYS reported "no boolean engine
available" -- every ``boolean={...}`` CSG request in ``build_mesh``
silently fell back to non-watertight concatenation, even with a real,
working boolean engine (``manifold3d``) installed. The tests below are
skipped (not failed) when the optional ``trimesh``/``manifold3d``
dependencies aren't installed, matching this repo's established
convention for CAD/geometry-format dependencies.
"""
from __future__ import annotations

import pytest

pytest.importorskip("trimesh")

from pinneapple_design.geometry.gen.primitives import (
    _boolean_engines_available,
    build_mesh,
    list_builders,
)


def test_list_builders_includes_all_known_primitive_aliases():
    names = list_builders()
    assert isinstance(names, tuple)
    for expected in ("box", "cube", "sphere", "cylinder", "plane", "channel"):
        assert expected in names


def test_boolean_engines_available_does_not_crash_and_returns_only_strings():
    """Regression test for the None/sorted() crash: whatever engines are
    actually installed on this machine, the function must return a
    tuple of plain strings (possibly empty), never raise."""
    engines = _boolean_engines_available()
    assert isinstance(engines, tuple)
    assert all(isinstance(e, str) for e in engines)


@pytest.mark.skipif(
    not _boolean_engines_available(),
    reason="no trimesh boolean engine (e.g. manifold3d) installed -- cannot verify a real watertight boolean",
)
def test_box_minus_cylinder_boolean_produces_a_real_watertight_solid_with_no_fallback_warning():
    """Regression test for the actual observed effect of the bug: before
    the fix, this exact call produced a UserWarning ("No trimesh boolean
    engine available... falling back to concatenate") and a
    non-watertight result, despite manifold3d being installed and
    perfectly capable of a real difference() here."""
    import warnings

    import trimesh

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        md = build_mesh(
            "box", extents=(1.0, 1.0, 1.0),
            boolean={"op": "cut", "other": {"name": "cylinder", "radius": 0.3, "height": 2.0}},
        )
        fallback_warnings = [w for w in caught if "falling back to concatenate" in str(w.message).lower()]
        assert not fallback_warnings, (
            f"boolean CSG silently fell back to non-watertight concatenation: {[str(w.message) for w in caught]}"
        )

    tm = trimesh.Trimesh(vertices=md.vertices, faces=md.faces, process=False)
    assert tm.is_watertight


def test_build_mesh_rejects_unknown_builder_name_with_a_clear_error():
    with pytest.raises(ValueError, match="Unsupported"):
        build_mesh("definitely_not_a_real_primitive_xyz")
