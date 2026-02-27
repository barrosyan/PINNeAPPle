from __future__ import annotations

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, InitialCondition
from ..typing import CoordNames


def laplace_2d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y")
    fields = ("u",)
    pde = PDETermSpec(kind="laplace", fields=fields, coords=coords, params={})
    bc = DirichletBC(
        name="u_boundary",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )
    return ProblemSpec(
        name="laplace_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc,),
        sample_defaults={"n_col": 50_000, "n_bc": 10_000},
        field_ranges={"u": (-1.0, 1.0)},
        references=("Laplace 2D template.",),
    )


def poisson_2d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "y")
    fields = ("u",)
    pde = PDETermSpec(kind="poisson", fields=fields, coords=coords, params={}, meta={"note": "Provide ctx['source_fn'] for f(x,y)."})
    bc = DirichletBC(
        name="u_boundary",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )
    return ProblemSpec(
        name="poisson_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc,),
        sample_defaults={"n_col": 60_000, "n_bc": 12_000},
        field_ranges={"u": (-2.0, 2.0)},
        references=("Poisson 2D template.",),
    )


def burgers_1d_default() -> ProblemSpec:
    coords: CoordNames = ("x", "t")
    fields = ("u",)
    pde = PDETermSpec(kind="burgers", fields=fields, coords=coords, params={"nu": 0.01})

    ic = InitialCondition(
        name="u_init",
        fields=("u",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 1], 0.0),
        value_fn=lambda X, ctx: np.sin(np.pi * X[:, 0:1]).astype(np.float32),
        weight=10.0,
    )

    return ProblemSpec(
        name="burgers_1d_default",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic,),
        sample_defaults={"n_col": 100_000, "n_ic": 20_000},
        field_ranges={"u": (-1.0, 1.0)},
        references=("1D viscous Burgers template.",),
    )