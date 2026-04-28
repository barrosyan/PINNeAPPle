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


def lid_driven_cavity_3d(
    Re: float = 100.0,
    size: float = 1.0,
    lid_velocity: float = 1.0,
) -> ProblemSpec:
    """3D Lid-Driven Cavity benchmark.

    Steady-state formulation on the unit cube [0, size]^3.  The lid face at
    z = size moves with velocity ``lid_velocity`` in the x-direction.  The
    remaining five faces are no-slip walls.

    Parameters
    ----------
    Re : float
        Reynolds number (= U_lid * size / nu).
    size : float
        Cube side length L.
    lid_velocity : float
        Lid speed U_lid.
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("u", "v", "w", "p")

    pde = PDETermSpec(
        kind="navier_stokes_incompressible",
        fields=fields,
        coords=coords,
        params={"Re": float(Re)},
        meta={"note": "Steady-state 3D LDC; BC tags: lid / walls."},
    )

    lid = DirichletBC(
        name="lid",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "lid"},
        value_fn=lambda X, ctx: np.column_stack([
            np.full(X.shape[0], float(lid_velocity), dtype=np.float32),
            np.zeros(X.shape[0], dtype=np.float32),
            np.zeros(X.shape[0], dtype=np.float32),
        ]),
        weight=50.0,
    )

    walls = DirichletBC(
        name="walls",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "walls"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="lid_driven_cavity_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(lid, walls),
        sample_defaults={"n_col": 200_000, "n_bc": 60_000},
        field_ranges={"u": (-2, 2), "v": (-2, 2), "w": (-2, 2), "p": (-2, 2)},
        references=("3D Lid-Driven Cavity Flow benchmark (Ghia et al. 1982 extension).",),
        domain_bounds={"x": (0.0, size), "y": (0.0, size), "z": (0.0, size)},
        solver_spec={
            "name": "fdm",
            "method": "lid_driven_cavity_3d",
            "params": {
                "nx": 32, "ny": 32, "nz": 32,
                "nu": 1.0 / Re,
                "dt": 5e-4,
                "nt": 500,
            },
        },
        meta={
            "description": "3D Lid-Driven Cavity Flow benchmark (Ghia et al. 1982 extension)",
            "digital_twin_fields": ["u", "v", "w", "p"],
            "benchmark_Re": Re,
        },
    )


def channel_flow_3d(
    Re: float = 100.0,
    length: float = 2.0,
    height: float = 1.0,
    width: float = 1.0,
    Umax: float = 1.0,
) -> ProblemSpec:
    """3D pressure-driven rectangular channel (Poiseuille) flow.

    Domain: [0, length] × [0, height] × [0, width].
    - Inlet at x=0: Poiseuille profile  u = 16·Umax·y(H-y)·z(W-z)/(H²W²).
    - Outlet at x=length: Neumann dp/dn = 0.
    - Four walls (y=0, y=H, z=0, z=W): no-slip.

    Parameters
    ----------
    Re : float
        Reynolds number.
    length : float
        Streamwise extent Lx.
    height : float
        Wall-normal extent H.
    width : float
        Spanwise extent W.
    Umax : float
        Maximum centreline velocity.
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("u", "v", "w", "p")

    H = float(height)
    W = float(width)
    _Umax = float(Umax)

    pde = PDETermSpec(
        kind="navier_stokes_incompressible",
        fields=fields,
        coords=coords,
        params={"Re": float(Re)},
        meta={"note": "3D rectangular channel (Poiseuille-like). BC tags: inlet/outlet/walls."},
    )

    # Inlet: Poiseuille profile for a rectangular duct
    def _inlet_profile(X: np.ndarray, ctx) -> np.ndarray:
        y = X[:, 1].astype(np.float32)
        z = X[:, 2].astype(np.float32)
        u = (16.0 * _Umax * y * (H - y) * z * (W - z) / (H ** 2 * W ** 2)).astype(np.float32)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        return np.stack([u, v, w], axis=1)

    inlet = DirichletBC(
        name="inlet",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=_inlet_profile,
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

    walls = DirichletBC(
        name="walls",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "walls"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="channel_flow_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(inlet, outlet, walls),
        sample_defaults={"n_col": 300_000, "n_bc": 80_000},
        field_ranges={"u": (-2, 2), "v": (-1, 1), "w": (-1, 1), "p": (-5, 5)},
        references=("3D rectangular duct Poiseuille flow.",),
        domain_bounds={"x": (0.0, length), "y": (0.0, height), "z": (0.0, width)},
        solver_spec={
            "name": "fdm",
            "method": "channel_flow_3d",
            "params": {
                "nx": 32, "ny": 32, "nz": 32,
                "nu": 1.0 / Re,
                "dt": 5e-4,
                "nt": 500,
            },
        },
        meta={
            "description": "3D pressure-driven rectangular channel (Poiseuille) flow",
            "digital_twin_fields": ["u", "v", "w", "p"],
            "Re": Re,
            "Umax": Umax,
        },
    )


def pipe_flow_3d(
    Re: float = 100.0,
    radius: float = 0.5,
    length: float = 2.0,
    Umax: float = 1.0,
) -> ProblemSpec:
    """3D Hagen-Poiseuille pipe flow in a circular cylinder.

    The pipe is a cylinder of radius ``radius`` aligned with the x-axis.
    The cross-section is centred at (y_c, z_c) = (0, 0) so the domain in
    y and z runs from -radius to +radius.

    - Inlet at x=0: Hagen-Poiseuille  u = 2·Umax·(1 - r²/R²).
    - Outlet at x=length: Neumann dp/dn = 0.
    - Cylindrical wall: no-slip (applied via a circular tag at r = radius).

    Parameters
    ----------
    Re : float
        Reynolds number (= 2·R·Umax / nu).
    radius : float
        Pipe radius R.
    length : float
        Pipe length L.
    Umax : float
        Centreline velocity.
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("u", "v", "w", "p")

    R = float(radius)
    _Umax = float(Umax)

    pde = PDETermSpec(
        kind="navier_stokes_incompressible",
        fields=fields,
        coords=coords,
        params={"Re": float(Re)},
        meta={"note": "3D Hagen-Poiseuille pipe flow. BC tags: inlet/outlet/wall."},
    )

    # Inlet: Hagen-Poiseuille profile
    def _hp_inlet(X: np.ndarray, ctx) -> np.ndarray:
        y = X[:, 1].astype(np.float64)
        z = X[:, 2].astype(np.float64)
        r2 = y ** 2 + z ** 2
        u = (2.0 * _Umax * np.maximum(0.0, 1.0 - r2 / R ** 2)).astype(np.float32)
        v = np.zeros_like(u)
        w = np.zeros_like(u)
        return np.stack([u, v, w], axis=1)

    inlet = DirichletBC(
        name="inlet",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=_hp_inlet,
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

    wall = DirichletBC(
        name="wall",
        fields=("u", "v", "w"),
        selector_type="tag",
        selector={"tag": "wall"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="pipe_flow_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(inlet, outlet, wall),
        sample_defaults={"n_col": 300_000, "n_bc": 80_000},
        field_ranges={"u": (-2, 2), "v": (-1, 1), "w": (-1, 1), "p": (-5, 5)},
        references=("3D Hagen-Poiseuille pipe flow.",),
        domain_bounds={"x": (0.0, length), "y": (-radius, radius), "z": (-radius, radius)},
        solver_spec={
            "name": "fdm",
            "method": "pipe_flow_3d",
            "params": {
                "nx": 32, "ny": 32, "nz": 32,
                "nu": 1.0 / Re,
                "dt": 5e-4,
                "nt": 500,
                "radius": radius,
                "length": length,
            },
        },
        meta={
            "description": "3D Hagen-Poiseuille pipe flow in a circular cylinder",
            "digital_twin_fields": ["u", "v", "w", "p"],
            "Re": Re,
            "radius": radius,
            "Umax": Umax,
        },
    )


def _register_3d_cfd_presets() -> None:
    """Register new 3-D CFD presets into the preset registry at import time."""
    try:
        from .registry import register_preset as _reg
        _reg("lid_driven_cavity_3d")(lid_driven_cavity_3d)
        _reg("channel_flow_3d")(channel_flow_3d)
        _reg("pipe_flow_3d")(pipe_flow_3d)
    except Exception:
        pass


_register_3d_cfd_presets()