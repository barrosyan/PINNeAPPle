"""Structural mechanics problem presets.

Covers:
- 2D plane stress / plane strain (linear elasticity)
- Von Mises stress formulation (derives von Mises from displacement fields)
- 3D linear elasticity (brackets, frames)
- Rotary coupling under combined torsion + axial load
- Thermoelasticity 2D/3D

Convention for elasticity:
  E  = Young's modulus (Pa)
  nu = Poisson's ratio
  lambda = E*nu / ((1+nu)*(1-2*nu))  (Lame first)
  mu     = E / (2*(1+nu))             (Lame second / shear modulus)
"""
from __future__ import annotations

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, NeumannBC
from ..scales import ScaleSpec
from ..environment_typing import CoordNames
from .registry import register_preset


def _lame(E: float, nu: float):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


@register_preset("plane_stress_2d")
def plane_stress_2d_default(
    E: float = 210e9,
    nu: float = 0.3,
    load_x: float = 0.0,
    load_y: float = -1000.0,
) -> ProblemSpec:
    """2D plane stress linear elasticity.

    Governing equations (strong form, body force neglected):
        sigma_xx_x + sigma_xy_y = 0
        sigma_xy_x + sigma_yy_y = 0
    where sigma = C : eps,  eps = sym(grad u)

    Fields: ux (x-displacement), uy (y-displacement)
    Derived (post-processing): von Mises stress = sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*sigma_xy^2)
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="linear_elasticity_plane_stress",
        fields=fields,
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={
            "formulation": "plane_stress",
            "note": "Fields: ux, uy. Von Mises computed post-hoc. Use ctx['body_force_fn'] for body forces.",
        },
    )

    fixed = DirichletBC(
        name_or_values="fixed",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=50.0,
    )

    traction_y = NeumannBC(
        name_or_values="traction_load",
        fields=("uy",),
        selector_type="tag",
        selector={"tag": "load"},
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), ctx.get("load_y", float(load_y)), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="plane_stress_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed, traction_y),
        sample_defaults={"n_col": 50_000, "n_bc": 15_000},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"ux": (-0.01, 0.01), "uy": (-0.01, 0.01)},
        references=("2D plane stress — Timoshenko & Goodier, Theory of Elasticity.",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        solver_spec={
            "name": "fenics",
            "formulation": "plane_stress",
            "params": {"mesh_size": 0.05, "degree": 2},
        },
    )


@register_preset("plane_strain_2d")
def plane_strain_2d_default(
    E: float = 210e9,
    nu: float = 0.3,
) -> ProblemSpec:
    """2D plane strain linear elasticity (deep structures, dams, tunnels)."""
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="linear_elasticity_plane_strain",
        fields=fields,
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={"formulation": "plane_strain"},
    )

    fixed = DirichletBC(
        name_or_values="fixed",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=50.0,
    )

    return ProblemSpec(
        name="plane_strain_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed,),
        sample_defaults={"n_col": 60_000, "n_bc": 20_000},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"ux": (-0.01, 0.01), "uy": (-0.01, 0.01)},
        references=("2D plane strain — Timoshenko & Goodier.",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        solver_spec={"name": "fenics", "formulation": "plane_strain", "params": {"mesh_size": 0.05, "degree": 2}},
    )


@register_preset("von_mises_2d")
def von_mises_2d_default(
    E: float = 210e9,
    nu: float = 0.3,
) -> ProblemSpec:
    """Plane stress with von Mises yield criterion as extra output field.

    Fields: ux, uy, vm (von Mises stress, derived).
    The PDE is still linear elasticity; vm is a post-processing output
    added as an extra field so surrogate models can predict it directly.
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy", "vm")  # vm = von Mises stress
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="linear_elasticity_plane_stress",
        fields=("ux", "uy"),  # PDE only on displacements
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={
            "formulation": "plane_stress_von_mises",
            "derived_fields": {"vm": "von_mises_stress"},
            "note": "vm is computed from ux, uy via Cauchy stress formula; not a direct PDE unknown.",
        },
    )

    fixed = DirichletBC(
        name_or_values="fixed",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=50.0,
    )

    return ProblemSpec(
        name="von_mises_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed,),
        sample_defaults={"n_col": 80_000, "n_bc": 20_000},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"ux": (-0.01, 0.01), "uy": (-0.01, 0.01), "vm": (0.0, 500e6)},
        references=("Von Mises yield criterion — von Mises (1913), Huber (1904).",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        solver_spec={"name": "fenics", "formulation": "plane_stress_von_mises", "params": {"mesh_size": 0.05, "degree": 2}},
    )


@register_preset("linear_elasticity_3d")
def linear_elasticity_3d(
    E: float = 210e9,
    nu: float = 0.3,
) -> ProblemSpec:
    """3D linear elasticity for brackets, frames, and general structures."""
    coords: CoordNames = ("x", "y", "z")
    fields = ("ux", "uy", "uz")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="linear_elasticity",
        fields=fields,
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={"formulation": "3d_elasticity"},
    )

    fixed = DirichletBC(
        name_or_values="fixed",
        fields=("ux", "uy", "uz"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=50.0,
    )

    traction = NeumannBC(
        name_or_values="traction",
        fields=("ux", "uy", "uz"),
        selector_type="tag",
        selector={"tag": "load"},
        value_fn=lambda X, ctx: np.tile([0.0, 0.0, ctx.get("load_z", -1000.0)], (X.shape[0], 1)).astype(np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="linear_elasticity_3d",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed, traction),
        sample_defaults={"n_col": 200_000, "n_bc": 60_000},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=1.0),
        field_ranges={"ux": (-0.05, 0.05), "uy": (-0.05, 0.05), "uz": (-0.05, 0.05)},
        references=("3D linear elasticity — Landau & Lifshitz, Theory of Elasticity.",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
        solver_spec={"name": "fenics", "formulation": "3d_elasticity", "params": {"mesh_size": 0.08, "degree": 2}},
    )


@register_preset("rotary_coupling_torsion")
def rotary_coupling_torsion_default(
    E: float = 210e9,
    nu: float = 0.3,
    r_inner: float = 0.05,
    r_outer: float = 0.08,
    torque: float = 10000.0,
    axial_force: float = 500000.0,
    length: float = 1.0,
) -> ProblemSpec:
    """Rotary coupling under combined torsion and axial tension.

    Geometry: hollow cylinder (pipe) of inner radius r_inner, outer radius r_outer,
    length `length`. Fixed at z=0, torque T and axial force F applied at z=length.

    Fields: ux, uy, uz (displacement components)
    Derived: von Mises stress, shear stress, axial stress

    Analytical solution (thin-walled):
      tau_max = T * r_outer / J,  J = pi/2 * (r_o^4 - r_i^4)
      sigma_z = F / A,            A = pi * (r_o^2 - r_i^2)
      vm = sqrt(sigma_z^2 + 3*tau^2)
    """
    coords: CoordNames = ("x", "y", "z")
    fields = ("ux", "uy", "uz")
    lam, mu = _lame(float(E), float(nu))

    # Torsion constant
    J = np.pi / 2 * (r_outer**4 - r_inner**4)
    # Cross-section area
    A = np.pi * (r_outer**2 - r_inner**2)
    tau_max = float(torque) * float(r_outer) / J
    sigma_z = float(axial_force) / A

    pde = PDETermSpec(
        kind="linear_elasticity",
        fields=fields,
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={
            "industry": "rotary_coupling",
            "r_inner": float(r_inner),
            "r_outer": float(r_outer),
            "torque": float(torque),
            "axial_force": float(axial_force),
            "analytical_tau_max": tau_max,
            "analytical_sigma_z": sigma_z,
            "analytical_vm": float(np.sqrt(sigma_z**2 + 3 * tau_max**2)),
        },
    )

    # Fixed at z=0 face: ux=uy=uz=0
    fixed = DirichletBC(
        name_or_values="fixed_end",
        fields=("ux", "uy", "uz"),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0, atol=1e-3),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 3), dtype=np.float32),
        weight=100.0,
    )

    def _torsion_traction(X, ctx):
        """Torsion (tangential) traction at z = length face: T = torque / J * r * e_theta"""
        x, y = X[:, 0], X[:, 1]
        r = np.sqrt(x**2 + y**2) + 1e-12
        tau = float(torque) / J * r
        # e_theta = (-y/r, x/r, 0)
        tx = -tau * y / r
        ty = tau * x / r
        tz = np.full_like(tx, float(axial_force) / A)
        return np.stack([tx, ty, tz], axis=1).astype(np.float32)

    traction_end = NeumannBC(
        name_or_values="torsion_traction",
        fields=("ux", "uy", "uz"),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], float(length), atol=1e-3),
        value_fn=_torsion_traction,
        weight=30.0,
    )

    inner_wall = DirichletBC(
        name_or_values="inner_wall_free",
        fields=("ux", "uy"),
        selector_type="callable",
        selector=lambda X, ctx: np.abs(np.sqrt(X[:, 0]**2 + X[:, 1]**2) - float(r_inner)) < 1e-3,
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=0.0,  # free surface, zero traction (not displacement BC)
    )

    return ProblemSpec(
        name="rotary_coupling_torsion_default",
        dim=3,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed, traction_end),
        sample_defaults={"n_col": 150_000, "n_bc": 50_000},
        scales=ScaleSpec(L=float(length), U=tau_max, alpha=1.0),
        field_ranges={"ux": (-0.001, 0.001), "uy": (-0.001, 0.001), "uz": (-0.005, 0.005)},
        references=(
            "Rotary coupling under combined torsion and axial load.",
            "Analytical solution: Timoshenko & Goodier Torsion of Circular Shafts.",
        ),
        domain_bounds={"x": (-r_outer, r_outer), "y": (-r_outer, r_outer), "z": (0.0, length)},
        solver_spec={
            "name": "fenics",
            "formulation": "3d_elasticity_cylinder",
            "params": {"mesh_size": 0.005, "degree": 2, "r_inner": r_inner, "r_outer": r_outer},
        },
    )


@register_preset("thermoelasticity_2d")
def thermoelasticity_2d_default(
    E: float = 70e9,
    nu: float = 0.33,
    alpha_T: float = 23e-6,
    dT: float = 100.0,
) -> ProblemSpec:
    """2D thermoelastic coupling: thermal expansion due to temperature change.

    PDE system:
      - Heat: div(k * grad T) = 0
      - Elasticity: div(sigma) = 0,  sigma = C:(eps - alpha_T * dT * I)

    Fields: ux, uy, T
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy", "T")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="thermoelasticity_2d",
        fields=fields,
        coords=coords,
        params={
            "E": float(E), "nu": float(nu), "lambda": lam, "mu": mu,
            "alpha_T": float(alpha_T), "dT_ref": float(dT),
        },
        meta={"formulation": "thermoelastic_plane_stress"},
    )

    fixed = DirichletBC(
        name_or_values="fixed",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "fixed"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=50.0,
    )

    T_hot = DirichletBC(
        name_or_values="T_hot",
        fields=("T",),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), float(dT), dtype=np.float32),
        weight=20.0,
    )

    T_cold = DirichletBC(
        name_or_values="T_cold",
        fields=("T",),
        selector_type="tag",
        selector={"tag": "outlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="thermoelasticity_2d_default",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(fixed, T_hot, T_cold),
        sample_defaults={"n_col": 80_000, "n_bc": 25_000},
        scales=ScaleSpec(L=1.0, U=1.0, alpha=float(alpha_T)),
        field_ranges={"ux": (-0.005, 0.005), "uy": (-0.005, 0.005), "T": (0.0, float(dT))},
        references=("Thermoelasticity — Boley & Weiner, Theory of Thermal Stresses.",),
        domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        solver_spec={"name": "fenics", "formulation": "thermoelastic_2d", "params": {"mesh_size": 0.05, "degree": 2}},
    )
