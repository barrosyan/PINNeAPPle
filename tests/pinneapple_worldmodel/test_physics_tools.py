"""Tests for pinneapple_worldmodel.physics_tools.PhysicsToolRegistry.

First test file for this module. Its main purpose is to guard against a
real regression found and fixed this session: 23 of the registry's 38
tools were permanently reported unavailable because ``_safe_wrap``'s
availability gate checked stale, never-existent top-level package names
(e.g. ``"pinneapple_uq"`` instead of the real ``"pinneapple_analysis"``)
while each tool's actual function body imported from the correct, real
module all along -- the tools were fully functional, just unreachable
through the registry. Fixed by correcting every ``_safe_wrap(...)``
module-name argument (and the parallel, purely-documentational
``module_path=`` field on each ``PhysicsTool``) to the real top-level
package the function body actually imports from.
"""
from __future__ import annotations

import importlib

import pytest

from pinneapple_worldmodel.physics_tools import PhysicsTool, PhysicsToolRegistry


@pytest.fixture(scope="module")
def registry() -> PhysicsToolRegistry:
    reg = PhysicsToolRegistry()
    reg.register_all()
    return reg


def test_registers_38_tools(registry):
    assert len(registry) == 38


def test_every_tools_module_name_is_a_real_top_level_package(registry):
    """The regression this file exists to prevent: a tool whose
    ``module_path`` (or, equivalently, whatever string was passed as the
    second argument to ``_safe_wrap``) names a top-level package that
    doesn't actually exist can never become available, no matter how
    correct its implementation is. Every registered tool's module_path's
    top-level component must be importable."""
    unresolvable = []
    for tool in registry._tools.values():
        top_level = tool.module_path.split(".")[0]
        try:
            importlib.import_module(top_level)
        except ImportError:
            unresolvable.append((tool.name, tool.module_path))
    assert unresolvable == [], f"tools with unresolvable module_path: {unresolvable}"


def test_all_tools_available_given_real_dependencies_installed(registry):
    """With every real pinneapple_* dependency installed in this
    environment (confirmed by the previous test), every tool should
    report available -- the whole point of the registry existing is that
    availability tracks REAL importability, not a hardcoded guess."""
    unavailable = [t.name for t in registry._tools.values() if not t.is_available()]
    assert unavailable == [], f"unexpectedly unavailable tools: {unavailable}"
    assert len(registry.available_tools()) == len(registry)


@pytest.mark.parametrize("tool_name,expected_module", [
    ("run_sph_simulation", "pinneapple_simulation"),
    ("compile_pinn", "pinneapple_physics"),
    ("identify_pde", "pinneapple_physics"),
    ("suggest_problem_spec", "pinneapple_physics"),
    ("validate_physics", "pinneapple_analysis"),
    ("mc_dropout_uncertainty", "pinneapple_analysis"),
    ("aleatoric_uncertainty", "pinneapple_analysis"),
    ("decompose_uncertainty", "pinneapple_analysis"),
    ("eki_parameter_inversion", "pinneapple_analysis"),
    ("sindy_equation_discovery", "pinneapple_analysis"),
    ("local_sensitivity", "pinneapple_analysis"),
    ("transfer_train", "pinneapple_adaptation"),
    ("parametric_family_transfer", "pinneapple_adaptation"),
    ("timeseries_forecast", "pinneapple_systems"),
    ("power_spectrum", "pinneapple_systems"),
    ("build_cosim", "pinneapple_systems"),
    ("run_cosim", "pinneapple_systems"),
    ("make_sdf", "pinneapple_design"),
    ("infer_on_grid_2d", "pinneapple_neural"),
    ("plot_physics_field", "pinneapple_neural"),
    ("bayesian_design_opt", "pinneapple_design"),
    ("build_digital_twin", "pinneapple_systems"),
    ("kalman_data_assimilation", "pinneapple_systems"),
])
def test_previously_broken_tool_now_available_with_correct_module(registry, tool_name, expected_module):
    """Each of the 23 tools found disabled by the stale-module-name bug,
    now individually confirmed both available and pointing at its real
    top-level package (not just "available" via some other accident)."""
    tool = registry.get(tool_name)
    assert tool.is_available(), f"{tool_name} should be available"
    assert tool.module_path.split(".")[0] == expected_module


def test_pure_function_tool_actually_executes_end_to_end():
    """Not just import-resolvable -- pick one previously-broken tool with
    no heavy external dependency (power spectrum of a plain array) and
    actually call it, to prove the fix restored real functionality, not
    just a passing availability flag."""
    import numpy as np

    reg = PhysicsToolRegistry()
    reg.register_all()
    tool = reg.get("power_spectrum")
    assert tool.is_available()
    data = np.sin(np.linspace(0, 20 * np.pi, 512)).astype("float32")
    import torch
    result = tool.call(data=torch.as_tensor(data))
    assert result is not None


def test_registry_get_unknown_tool_raises_keyerror(registry):
    with pytest.raises(KeyError):
        registry.get("this_tool_does_not_exist")


def test_summary_reports_full_availability(registry):
    text = registry.summary()
    assert "38 tools" in text
    assert "(38 available)" in text


def test_physics_tool_call_without_fn_raises_runtime_error():
    tool = PhysicsTool(
        name="unregistered", category="test", description="",
        input_schema={}, output_schema={}, module_path="nonexistent_module",
        fn=None,
    )
    assert not tool.is_available()
    with pytest.raises(RuntimeError):
        tool.call()
