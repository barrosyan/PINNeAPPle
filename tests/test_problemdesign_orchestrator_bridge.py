"""Tests for the additive, one-way, read-only bridge from
pinneapple_problemdesign's plan generation to pinneapple_worldmodel's
PhysicsOrchestrator tool registry.

Covers:
  (a) existing build_plan / build_plan_cfd_first / build_plan_pinn_first /
      build_plan_fno_first callers are unaffected -- identical output --
      when the bridge is disabled, and when pinneapple_worldmodel is made
      unimportable (simulated for real via sys.modules / import hook, not
      just assumed).
  (b) with the real pinneapple_worldmodel.physics_tools.PhysicsToolRegistry
      actually available, a CFD-domain ProblemSpec produces a plan whose
      extra step names at least one real, currently-registered tool --
      asserted against what the live registry actually returns right now.
  (c) no circular import: pinneapple_worldmodel does not import anything
      from pinneapple_problemdesign.
"""
from __future__ import annotations

import builtins
import copy
import importlib
import subprocess
import sys

import pytest

from pinneapple_problemdesign.schema import ProblemSpec, PhysicsSpec, GeometrySpec
from pinneapple_problemdesign.knowledge.mapping import (
    available_orchestrator_tools,
    build_plan,
    build_plan_cfd_first,
    build_plan_fno_first,
    build_plan_pinn_first,
)


def _cfd_spec() -> ProblemSpec:
    return ProblemSpec(
        title="Airfoil AoA sweep",
        goal="Predict lift and drag coefficients for an airfoil across an angle of attack sweep.",
        task_type="other",
        domain_context="external aerodynamics wind tunnel study",
        physics=PhysicsSpec(numerical_method="LBM"),
        geometry=GeometrySpec(aoa_sweep_deg=[0.0, 5.0, 10.0, 15.0]),
    )


def _pinn_spec() -> ProblemSpec:
    return ProblemSpec(
        title="Heat conduction",
        goal="Solve 2D heat conduction PDE",
        task_type="pde_solution",
    )


def _fno_spec() -> ProblemSpec:
    return ProblemSpec(
        title="Sensor forecasting",
        goal="Forecast sensor readings",
        task_type="forecasting",
    )


# ---------------------------------------------------------------------------
# (a) Non-breaking: identical output when bridge disabled or worldmodel absent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec_factory", [_cfd_spec, _pinn_spec, _fno_spec])
def test_build_plan_bridge_disabled_matches_sub_builder(spec_factory):
    """With use_orchestrator_bridge=False, build_plan's output must be
    byte-for-byte identical to the underlying static sub-builder's output --
    i.e. the new capability is fully opt-out-able and additive-only."""
    spec = spec_factory()
    gaps = []

    plan_via_dispatch = build_plan(spec, gaps, use_orchestrator_bridge=False)

    if spec.task_type == "pde_solution":
        plan_direct = build_plan_pinn_first(spec, gaps)
    elif spec.task_type == "forecasting":
        plan_direct = build_plan_fno_first(spec, gaps)
    else:
        plan_direct = build_plan_cfd_first(spec, gaps)

    assert plan_via_dispatch == plan_direct


def test_sub_builders_never_touched_by_bridge():
    """build_plan_cfd_first / build_plan_pinn_first / build_plan_fno_first
    are the pre-existing static builders and must never gain the extra
    orchestrator-tools step themselves -- only the build_plan() dispatcher
    appends it. This guarantees direct callers of the sub-builders (e.g.
    examples/problem_designer/03_offline_spec_to_report.py) see zero
    behavior change."""
    cfd_plan = build_plan_cfd_first(_cfd_spec(), [])
    pinn_plan = build_plan_pinn_first(_pinn_spec(), [])
    fno_plan = build_plan_fno_first(_fno_spec(), [])

    for plan in (cfd_plan, pinn_plan, fno_plan):
        titles = [s.title for s in plan.steps]
        assert "Available orchestrator tools" not in titles


def test_build_plan_default_is_deterministic_and_backward_compatible_shape():
    """Calling build_plan with the old two-positional-argument signature
    (no use_orchestrator_bridge passed) must still work -- the new
    parameter is purely additive with a default."""
    spec = _pinn_spec()
    gaps = []
    plan = build_plan(spec, gaps)  # old call signature, unchanged
    assert plan.recommended_approach  # sanity: still produces a real plan
    assert plan.steps  # still has the original PINN-first steps


def test_bridge_returns_empty_list_when_worldmodel_import_fails(monkeypatch):
    """Simulate pinneapple_worldmodel being genuinely unimportable (optional
    dependency absent) by making the import machinery raise ImportError for
    it, rather than merely assuming the try/except works."""
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "pinneapple_worldmodel" or name.startswith("pinneapple_worldmodel."):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, *args, **kwargs)

    # Also remove any already-imported worldmodel modules from the module
    # cache so the blocked __import__ is actually exercised.
    removed = {}
    for mod_name in list(sys.modules):
        if mod_name == "pinneapple_worldmodel" or mod_name.startswith("pinneapple_worldmodel."):
            removed[mod_name] = sys.modules.pop(mod_name)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    try:
        tools = available_orchestrator_tools(_cfd_spec())
        assert tools == []

        # And build_plan() with the bridge enabled must fall back gracefully
        # to exactly the static plan (no extra step, no exception).
        plan = build_plan(_cfd_spec(), [], use_orchestrator_bridge=True)
        static_plan = build_plan_cfd_first(_cfd_spec(), [])
        assert plan == static_plan
    finally:
        # Restore any modules we evicted so we don't poison later tests.
        sys.modules.update(removed)


def test_bridge_survives_registry_construction_error(monkeypatch):
    """If PhysicsToolRegistry() or register_all() raises for any other
    (non-ImportError) reason, the bridge must still degrade to []
    rather than propagating the exception."""
    import pinneapple_worldmodel.physics_tools as physics_tools_mod

    class _ExplodingRegistry:
        def __init__(self):
            raise RuntimeError("simulated construction failure")

    monkeypatch.setattr(physics_tools_mod, "PhysicsToolRegistry", _ExplodingRegistry)

    tools = available_orchestrator_tools(_cfd_spec())
    assert tools == []


# ---------------------------------------------------------------------------
# (b) With the real registry available, CFD spec surfaces real live tools
# ---------------------------------------------------------------------------

def test_bridge_surfaces_real_live_tools_for_cfd_spec():
    pytest.importorskip("pinneapple_worldmodel")
    from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry

    spec = _cfd_spec()
    tools = available_orchestrator_tools(spec)

    assert tools, "expected at least one real orchestrator tool for a CFD-domain spec"

    # Cross-check against what the live registry actually reports right now,
    # so this test catches drift if tools are renamed/removed later instead
    # of relying on a hardcoded/mocked tool list.
    reg = PhysicsToolRegistry()
    reg.register_all()
    live_names = set()
    for category in ("simulation", "pde_solving", "geometry"):
        for t in reg.list_by_category(category):
            if t.is_available():
                live_names.add(t.name)

    returned_names = {t["name"] for t in tools}
    assert returned_names, "bridge returned tools with no names"
    assert returned_names <= live_names
    # Every tool the bridge found must actually be gettable from the live
    # registry (i.e. real, not invented).
    for name in returned_names:
        real_tool = reg.get(name)
        assert real_tool.is_available()


def test_build_plan_cfd_spec_includes_orchestrator_tools_step():
    pytest.importorskip("pinneapple_worldmodel")
    spec = _cfd_spec()
    plan = build_plan(spec, [], use_orchestrator_bridge=True)

    static_plan = build_plan_cfd_first(spec, [])
    extra_steps = plan.steps[len(static_plan.steps):]

    assert len(plan.steps) == len(static_plan.steps) + 1
    assert len(extra_steps) == 1
    extra = extra_steps[0]
    assert extra.title == "Available orchestrator tools"
    assert extra.actions, "expected at least one real tool named in actions"

    # The rest of the plan (recommended_approach, alternatives, the original
    # static steps, go_no_go) must be byte-for-byte unchanged -- additive
    # only, nothing removed or altered.
    assert plan.recommended_approach == static_plan.recommended_approach
    assert plan.alternatives == static_plan.alternatives
    assert plan.go_no_go == static_plan.go_no_go
    assert plan.steps[: len(static_plan.steps)] == static_plan.steps


def test_available_orchestrator_tools_metadata_shape():
    pytest.importorskip("pinneapple_worldmodel")
    tools = available_orchestrator_tools(_cfd_spec())
    assert tools
    for t in tools:
        assert set(t.keys()) == {"name", "category", "description", "module_path", "tags"}
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t["category"], str) and t["category"]
        assert isinstance(t["tags"], list)


def test_no_relevant_tools_means_no_placeholder_step():
    """If the (real or simulated) registry has nothing relevant, build_plan
    must not append a fake/placeholder step -- the extra step is simply
    absent."""
    spec = _fno_spec()
    plan_no_bridge = build_plan(spec, [], use_orchestrator_bridge=False)
    tools = available_orchestrator_tools(spec)
    plan_with_bridge = build_plan(spec, [], use_orchestrator_bridge=True)

    if not tools:
        assert plan_with_bridge == plan_no_bridge
    else:
        # If the live registry does have relevant tools for this spec, the
        # extra step must be genuinely populated, not an empty placeholder.
        assert len(plan_with_bridge.steps) == len(plan_no_bridge.steps) + 1
        assert plan_with_bridge.steps[-1].actions


# ---------------------------------------------------------------------------
# (c) No circular import
# ---------------------------------------------------------------------------

def test_worldmodel_does_not_import_problemdesign():
    """Grep pinneapple_worldmodel's source for any reference to
    pinneapple_problemdesign -- the dependency direction must stay strictly
    one-way (problemdesign -> optionally worldmodel, never the reverse)."""
    result = subprocess.run(
        ["grep", "-rl", "pinneapple_problemdesign", "pinneapple_worldmodel/"],
        cwd="/Users/yanbarros/Documents/GitHub/PINNeAPPle",
        capture_output=True,
        text=True,
    )
    # grep exit code 1 == no matches found (what we want); 0 == found matches (fail).
    assert result.returncode != 0, (
        f"pinneapple_worldmodel references pinneapple_problemdesign in: {result.stdout}"
    )


def test_worldmodel_importable_without_problemdesign_loaded():
    """Import pinneapple_worldmodel.physics_tools in a fresh subprocess with
    pinneapple_problemdesign never imported, proving worldmodel has no
    hidden dependency on it."""
    code = (
        "import sys\n"
        "assert 'pinneapple_problemdesign' not in sys.modules\n"
        "from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry\n"
        "reg = PhysicsToolRegistry(); reg.register_all()\n"
        "assert 'pinneapple_problemdesign' not in sys.modules\n"
        "print('OK', len(reg))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/Users/yanbarros/Documents/GitHub/PINNeAPPle",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
