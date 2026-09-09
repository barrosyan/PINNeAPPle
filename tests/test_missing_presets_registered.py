"""Regression test for previously-unregistered problem presets.

``ns_incompressible_2d_default`` in
``pinneapple_physics/pde_environment/presets/cfd.py`` was a real, complete
preset factory (same shape as its already-registered 3D siblings in the same
file) but was missing the ``@register_preset(...)`` decorator, so it was
invisible to ``list_presets()``/``get_preset()`` despite being referenced
elsewhere (e.g. the registry module's own docstring). This test locks in the
fix.

Note: ``heat_1d``, ``heat_2d``, and ``wave_1d`` (referenced in
``examples/use_cases/TEMPLATE/template_config.yaml``) are NOT covered here.
No real factory implementation exists anywhere in the codebase under those
names or an obvious 1D/2D variant -- registering them would require
authoring new preset physics, which is out of scope for a registration-gap
fix. (The template's own "heat_2d / poisson_2d" comment suggests the
already-registered ``poisson_2d`` preset is the intended real analog for
"2D steady-state heat"; ``heat_1d`` and ``wave_1d`` have no analog at all.)
"""
from __future__ import annotations

from pinneapple_physics.pde_environment.presets.registry import get_preset, list_presets
from pinneapple_physics.pde_environment.spec import ProblemSpec


def test_ns_incompressible_2d_is_registered():
    names = list_presets()
    assert "ns_incompressible_2d" in names


def test_ns_incompressible_2d_returns_valid_spec():
    spec = get_preset("ns_incompressible_2d", Re=200.0)

    assert isinstance(spec, ProblemSpec)
    assert spec.dim == 2
    assert spec.coords == ("x", "y", "t")
    assert spec.fields == ("u", "v", "p")

    assert spec.pde.kind == "navier_stokes_incompressible"
    assert spec.pde.params.get("Re") == 200.0

    # Boundary conditions: walls, inlet, outlet (channel-like NS setup).
    assert len(spec.conditions) == 3
    cond_names = {c.name for c in spec.conditions}
    assert {"walls", "inlet", "outlet_dp_dn"} <= cond_names

    # Default Re (not overridden) should differ from the explicit Re above.
    default_spec = get_preset("ns_incompressible_2d")
    assert default_spec.pde.params.get("Re") == 100.0
