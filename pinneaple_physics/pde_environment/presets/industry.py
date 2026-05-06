from __future__ import annotations

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, InitialCondition
from ..scales import ScaleSpec
from ..environment_typing import CoordNames


def steady_heat_conduction_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z")
    fields = ("T",)

    pde = PDETermSpec(
        kind="poisson",
        fields=fields,
        coords=coords,
        params={},
        meta={"industry": "thermal", "note": "Provide ctx['source_fn'] for volumetric heat generation (q/k)."},
    )

    bc_boundary = DirichletBC(
        name_or_values="T_boundary",
        fields=("T",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    # Example heater on inlet plane (hot patch). User can override by changing preset or ctx.
    bc_inlet_hot = DirichletBC(
        name_or_values="T_inlet_hot",
        fields=("T",),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="steady_heat_conduction_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc_boundary, bc_inlet_hot),
        sample_defaults={"n_col": 120_000, "n_bc": 40_000, "n_ic": 0, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"T": (0.0, 1.0)},
        references=("Industrial steady heat conduction (electronics, furnaces, heat sinks).",),
    )


def transient_heat_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z", "t")
    fields = ("T",)

    alpha = 1e-3
    pde = PDETermSpec(
        kind="heat_equation",
        fields=fields,
        coords=coords,
        params={"alpha": float(alpha)},
        meta={"industry": "thermal", "note": "Provide ctx['source_fn'] for heat source q(x,t)."},
    )

    ic = InitialCondition(
        name_or_values="T_init",
        fields=("T",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 3], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    bc = DirichletBC(
        name_or_values="T_boundary",
        fields=("T",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    return ProblemSpec(
        name="transient_heat_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic, bc),
        sample_defaults={"n_col": 200_000, "n_bc": 60_000, "n_ic": 30_000, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=float(alpha)),
        field_ranges={"T": (0.0, 1.0)},
        references=("Industrial transient heat (quenching, heating cycles, thermal shock).",),
    )


def linear_elasticity_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z")
    fields = ("ux", "uy", "uz")

    lam = 1.0
    mu = 1.0
    pde = PDETermSpec(
        kind="linear_elasticity",
        fields=fields,
        coords=coords,
        params={"lambda": float(lam), "mu": float(mu)},
        meta={"industry": "solid_mechanics", "note": "Use ctx['body_force_fn'] to provide b(x)."},
    )

    fixed = DirichletBC(
        name_or_values="fixed_support",
        fields=("ux", "uy", "uz"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=50.0,
    )

    return ProblemSpec(
        name="linear_elasticity_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed,),
        sample_defaults={"n_col": 220_000, "n_bc": 80_000, "n_ic": 0, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"ux": (-0.05, 0.05), "uy": (-0.05, 0.05), "uz": (-0.05, 0.05)},
        references=("Industrial linear elasticity (brackets, frames, stress analysis).",),
    )


def darcy_pressure_only_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z")
    fields = ("p",)

    k = 1.0
    pde = PDETermSpec(
        kind="darcy",
        fields=fields,
        coords=coords,
        params={"k": float(k), "mu": 1.0},
        meta={"mode": "pressure_only", "industry": "porous_media", "note": "Provide ctx['source_fn']=s(x) if needed."},
    )

    bc_in = DirichletBC(
        name_or_values="p_inlet",
        fields=("p",),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    bc_out = DirichletBC(
        name_or_values="p_outlet",
        fields=("p",),
        selector_type="tag",
        selector={"tag": "outlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="darcy_pressure_only_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc_in, bc_out),
        sample_defaults={"n_col": 180_000, "n_bc": 50_000, "n_ic": 0, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"p": (-1.0, 2.0)},
        references=("Darcy flow (reservoir simulation, groundwater).",),
    )


def helmholtz_acoustics_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z")
    fields = ("u",)

    k = 10.0
    pde = PDETermSpec(
        kind="helmholtz",
        fields=fields,
        coords=coords,
        params={"k": float(k)},
        meta={"industry": "acoustics", "note": "Provide ctx['source_fn']=f(x) if needed."},
    )

    bc = DirichletBC(
        name_or_values="u_boundary",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    return ProblemSpec(
        name="helmholtz_acoustics_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc,),
        sample_defaults={"n_col": 220_000, "n_bc": 70_000, "n_ic": 0, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"u": (-1.0, 1.0)},
        references=("Helmholtz (acoustic cavities, resonators).",),
    )


def wave_ultrasound_3d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "z", "t")
    fields = ("u",)

    c = 1.0
    pde = PDETermSpec(
        kind="wave_equation",
        fields=fields,
        coords=coords,
        params={"c": float(c)},
        meta={"industry": "ultrasound_vibration", "note": "Provide ctx['source_fn']=f(x,t) if needed."},
    )

    ic_u = InitialCondition(
        name_or_values="u_init",
        fields=("u",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 3], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    bc = DirichletBC(
        name_or_values="u_boundary",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    return ProblemSpec(
        name="wave_ultrasound_3d_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_u, bc),
        sample_defaults={"n_col": 260_000, "n_bc": 80_000, "n_ic": 30_000, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"u": (-1.0, 1.0)},
        references=("Wave equation (ultrasound testing, vibration).",),
    )


def reaction_diffusion_2d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y", "t")
    fields = ("c",)

    D = 1e-3
    pde = PDETermSpec(
        kind="heat_equation",
        fields=fields,
        coords=coords,
        params={"alpha": float(D)},
        meta={"industry": "chemistry", "note": "Use ctx['source_fn'] to implement reaction R(c)."},
    )

    ic = InitialCondition(
        name_or_values="c_init",
        fields=("c",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    bc = DirichletBC(
        name_or_values="c_boundary",
        fields=("c",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=1.0,
    )

    return ProblemSpec(
        name="reaction_diffusion_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic, bc),
        sample_defaults={"n_col": 160_000, "n_bc": 30_000, "n_ic": 20_000, "n_data": 0},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=float(D)),
        field_ranges={"c": (0.0, 1.0)},
        references=("Reaction–diffusion template for industrial chemistry / materials.",),
    )