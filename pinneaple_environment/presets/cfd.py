from __future__ import annotations

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, NeumannBC
from ..typing import CoordNames


def _unit_interval(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (x - a) / (b - a + 1e-12)


def _parabolic_profile_2d(X: np.ndarray, ctx, Umax: float = 1.0) -> np.ndarray:
    # assumes channel is aligned with x (flow direction), profile over y in [0,1]
    b = ctx.get("bounds", {})
    bmin = np.asarray(b.get("min", [-1, -1, -1]), dtype=np.float32)
    bmax = np.asarray(b.get("max", [1, 1, 1]), dtype=np.float32)
    y = _unit_interval(X[:, 1], float(bmin[1]), float(bmax[1]))
    u = 4.0 * Umax * y * (1.0 - y)
    v = np.zeros_like(u)
    return np.stack([u, v], axis=1).astype(np.float32)


def _poiseuille_profile_3d(X: np.ndarray, ctx, Umax: float = 1.0) -> np.ndarray:
    # simple separable profile over (y,z) in [0,1]^2
    b = ctx.get("bounds", {})
    bmin = np.asarray(b.get("min", [-1, -1, -1]), dtype=np.float32)
    bmax = np.asarray(b.get("max", [1, 1, 1]), dtype=np.float32)
    y = _unit_interval(X[:, 1], float(bmin[1]), float(bmax[1]))
    z = _unit_interval(X[:, 2], float(bmin[2]), float(bmax[2]))
    u = 16.0 * Umax * y * (1.0 - y) * z * (1.0 - z)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    return np.stack([u, v, w], axis=1).astype(np.float32)


def ns_incompressible_2d_default(Re: float = 100.0, Umax: float = 1.0) -> ProblemSpec:
    coords: CoordNames = ("x", "y", "t")
    fields = ("u", "v", "p")
    pde = PDETermSpec(
        kind="navier_stokes_incompressible",
        fields=fields,
        coords=coords,
        params={"Re": float(Re)},
        meta={"note": "Requires tags: inlet/outlet/walls. Inlet uses parabolic profile; outlet uses Neumann dp/dn=0 by default."},
    )

    walls = DirichletBC(
        name="walls",
        fields=("u", "v"),
        selector_type="tag",
        selector={"tag": "walls"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=20.0,
    )

    inlet = DirichletBC(
        name="inlet",
        fields=("u", "v"),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: _parabolic_profile_2d(X, ctx, Umax=Umax),
        weight=30.0,
    )

    # outlet: dp/dn = 0 (more stable than hard pressure for many cases)
    outlet = NeumannBC(
        name="outlet_dp_dn",
        fields=("p",),
        selector_type="tag",
        selector={"tag": "outlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="ns_incompressible_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(walls, inlet, outlet),
        sample_defaults={"n_col": 250_000, "n_bc": 70_000},
        field_ranges={"u": (-2, 2), "v": (-2, 2), "p": (-2, 2)},
        references=("Incompressible NS 2D template (channel-like).",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)},
        solver_spec={"name": "fdm", "method": "ns_projection_2d", "params": {"nx": 64, "ny": 64, "Re": 100.0, "dt": 0.001, "t_end": 0.5}},
    )


def ns_incompressible_3d_default(Re: float = 100.0, Umax: float = 1.0) -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z", "t")
    fields = ("u", "v", "w", "p")
    pde = PDETermSpec(
        kind="navier_stokes_incompressible",
        fields=fields,
        coords=coords,
        params={"Re": float(Re)},
        meta={"note": "Requires tags: inlet/outlet/walls. Inlet uses Poiseuille-like profile; outlet uses Neumann dp/dn=0 by default."},
    )

    walls = DirichletBC(
        name="walls",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "walls"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=20.0,
    )

    inlet = DirichletBC(
        name="inlet",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: _poiseuille_profile_3d(X, ctx, Umax=Umax),
        weight=30.0,
    )

    outlet = NeumannBC(
        name="outlet_dp_dn",
        fields=("p",),
        selector_type="tag",
        selector={"tag": "outlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="ns_incompressible_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(walls, inlet, outlet),
        sample_defaults={"n_col": 350_000, "n_bc": 100_000},
        field_ranges={"u": (-2, 2), "v": (-2, 2), "w": (-2, 2), "p": (-2, 2)},
        references=("Incompressible NS 3D template (duct-like).",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0), "t": (0.0, 1.0)},
        solver_spec={"name": "fdm", "method": "ns_projection_3d", "params": {"nx": 32, "ny": 32, "nz": 32, "Re": 100.0, "dt": 0.001, "t_end": 0.5}},
    )