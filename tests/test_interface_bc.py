"""Tests for InterfaceBC: a multi-domain coupling boundary condition."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pinneapple_physics.pde_environment.conditions import ConditionSpec, InterfaceBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem


def test_interface_bc_constructs_correctly():
    cond = InterfaceBC(
        "solid_fluid_interface",
        ("T_solid", "T_fluid"),
        selector_type="tag",
        selector={"tag": "chr_interface"},
        weight=5.0,
        k_a=55.0,
        k_b=0.6,
    )
    assert isinstance(cond, ConditionSpec)
    assert cond.kind == "interface"
    assert cond.fields == ("T_solid", "T_fluid")
    assert cond.name == "solid_fluid_interface"
    assert cond.weight == 5.0
    assert cond.interface_coeffs == {"k_a": 55.0, "k_b": 0.6}


def test_interface_bc_default_coeffs_are_unity():
    cond = InterfaceBC("iface", ("u", "v"))
    assert cond.interface_coeffs == {"k_a": 1.0, "k_b": 1.0}


def test_interface_bc_requires_exactly_two_fields():
    with pytest.raises(ValueError):
        InterfaceBC("bad", ("only_one",))
    with pytest.raises(ValueError):
        InterfaceBC("bad", ("a", "b", "c"))


def test_interface_bc_default_value_fn_gives_zero_continuity_target():
    cond = InterfaceBC("iface", ("u", "v"))
    X = np.zeros((4, 2), dtype=np.float32)
    vals = cond.values(X, {})
    assert vals.shape == (4, 2)
    assert np.allclose(vals, 0.0)


# ---------------------------------------------------------------------------
# End-to-end: a minimal two-region toy PDE using InterfaceBC through the
# real compiler (pinneapple_physics/pinn_solver/compiler/compile.py), which
# is where a ConditionSpec is actually turned into a loss term.
# ---------------------------------------------------------------------------


def _toy_model_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    n_col, n_bc = 64, 16

    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))

    x_col = torch.rand(n_col, 2)
    x_bc = torch.rand(n_bc, 2)
    n_bc_normals = torch.zeros(n_bc, 2)
    n_bc_normals[:, 0] = 1.0  # unit normal along x

    batch = {
        "x_col": x_col,
        "x_bc": x_bc,
        "y_bc": None,
        "n_bc": n_bc_normals,
        "ctx": {},
    }
    return model, batch


def _toy_spec(k_a: float = 1.0, k_b: float = 1.0) -> ProblemSpec:
    # Two-field toy PDE (reaction_diffusion supports exactly fields u,v
    # with a time-like coord) standing in for "field on side A" / "field
    # on side B" of a coupled interface, e.g. a conjugate heat-transfer
    # setup with u=T_solid, v=T_fluid.
    coords = ("x", "t")
    fields = ("u", "v")
    pde = PDETermSpec(kind="reaction_diffusion", fields=fields, coords=coords, params={})
    interface = InterfaceBC("region_interface", ("u", "v"), weight=5.0, k_a=k_a, k_b=k_b)
    return ProblemSpec(
        name="toy_interface_problem",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(interface,),
    )


def test_interface_bc_builds_and_produces_loss_term_via_real_compiler():
    spec = _toy_spec()
    loss_fn = compile_problem(spec)
    model, batch = _toy_model_and_batch()

    out = loss_fn(model, None, batch)

    assert "bc_region_interface" in out
    interface_loss = out["bc_region_interface"]
    assert interface_loss.item() > 0.0
    assert "total" in out
    assert torch.isfinite(out["total"])


def test_interface_bc_loss_term_is_differentiable():
    spec = _toy_spec(k_a=2.0, k_b=0.5)
    loss_fn = compile_problem(spec)
    model, batch = _toy_model_and_batch(seed=1)

    out = loss_fn(model, None, batch)
    total = out["total"]
    total.backward()

    grads = [p.grad for p in model.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)
