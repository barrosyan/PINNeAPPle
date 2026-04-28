"""Solid mechanics presets — elasticity, axisymmetric, contact problems.

Covers problems that arise in structural / mechanical engineering:
  - Axisymmetric linear elasticity (hollow cylinders, pressure vessels,
    threaded couplings, shafts, valve seats, …)
  - Thick-walled cylinder under pressure (Lamé — analytical solution available)
  - Parametric threaded connection (e.g. TC50 pin-box coupling)
  - 2D plane strain for cross-sections under plane loading

Convention
----------
  E  : Young's modulus (Pa)
  nu : Poisson's ratio
  lambda = E·nu / ((1+nu)·(1-2ν))   — Lamé first parameter
  mu     = E / (2·(1+nu))             — Lamé second / shear modulus

Axisymmetric (r, z) strain–displacement relations
--------------------------------------------------
  ε_rr = ∂u_r/∂r
  ε_zz = ∂u_z/∂z
  ε_θθ = u_r / r            ← hoop strain
  ε_rz = (∂u_r/∂z + ∂u_z/∂r) / 2

Constitutive (isotropic, small strain)
---------------------------------------
  σ_rr = λ·tr + 2μ·ε_rr
  σ_zz = λ·tr + 2μ·ε_zz
  σ_θθ = λ·tr + 2μ·ε_θθ    (tr = ε_rr + ε_zz + ε_θθ)
  σ_rz = 2μ·ε_rz

Equilibrium (body force = 0)
-----------------------------
  ∂σ_rr/∂r + ∂σ_rz/∂z + (σ_rr − σ_θθ)/r = 0
  ∂σ_rz/∂r + ∂σ_zz/∂z + σ_rz/r          = 0
"""
from __future__ import annotations

import numpy as np

from ..conditions import DirichletBC, NeumannBC, DataConstraint, ConditionSpec
from ..scales import ScaleSpec
from ..spec import PDETermSpec, ProblemSpec
from ..environment_typing import CoordNames
from .registry import register_preset


def _lame(E: float, nu: float):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return float(lam), float(mu)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — General axisymmetric linear elasticity
# ══════════════════════════════════════════════════════════════════════════════

@register_preset("axisymmetric_linear_elasticity_2d")
def axisymmetric_linear_elasticity_2d_default(
    r_min: float = 10.0,
    r_max: float = 50.0,
    z_min: float = 0.0,
    z_max: float = 100.0,
    E: float = 2.1e11,
    nu: float = 0.3,
    p_inner: float = 0.0,        # Pa — internal pressure (Neumann at r=r_min)
    p_outer: float = 0.0,        # Pa — external pressure (Neumann at r=r_max)
    traction_z_top: float = 0.0, # Pa — axial traction at z=z_max
    fix_z_bottom: bool = True,   # u_z = 0 at z=z_min
    fix_r_inner: bool = False,   # u_r = 0 at r=r_min  (symmetry axis if r_min=0)
) -> ProblemSpec:
    """
    2D axisymmetric linear elasticity in (r, z) coordinates.

    Fields : u_r (radial displacement), u_z (axial displacement)
    PDE    : axisymmetric equilibrium — div(σ) = 0 in cylindrical coords
    Domain : r ∈ [r_min, r_max], z ∈ [z_min, z_max]

    Boundary conditions (all optional, controlled by parameters):
      - u_z = 0 at z = z_min  (fixed base)
      - u_r = 0 at r = r_min  (axis of symmetry OR rigid bore)
      - σ_rr = p_inner at r = r_min (internal pressure)
      - σ_rr = p_outer at r = r_max (external pressure)
      - σ_zz = traction_z_top at z = z_max (axial traction)

    Analytical solution available for thick-walled cylinder:
      Use get_preset("thick_walled_cylinder_lame", ...)

    Applications
    ------------
    Threaded couplings, pressure vessels, gun barrels, rotating shafts,
    valve seats, bearing races, bolted flanges.
    """
    coords: CoordNames = ("r", "z")
    fields = ("u_r", "u_z")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="axisymmetric_linear_elasticity",
        fields=fields,
        coords=coords,
        params={
            "E": float(E), "nu": float(nu),
            "lambda": lam, "mu": mu,
        },
        meta={
            "formulation": "axisymmetric_rz",
            "equilibrium": [
                "d(s_rr)/dr + d(s_rz)/dz + (s_rr - s_tt)/r = 0",
                "d(s_rz)/dr + d(s_zz)/dz + s_rz/r = 0",
            ],
            "strain": {
                "e_rr": "du_r/dr",
                "e_zz": "du_z/dz",
                "e_tt": "u_r/r",
                "e_rz": "(du_r/dz + du_z/dr)/2",
            },
        },
    )

    conditions = []

    # Fixed base: u_z = 0 at z = z_min
    if fix_z_bottom:
        def _zero_uz(X, ctx):
            return np.zeros((X.shape[0], 1), dtype=np.float32)

        conditions.append(DirichletBC(
            name="fixed_base_uz",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _zmin=float(z_min): np.abs(X[:, 1] - _zmin) < 1e-6,
            value_fn=_zero_uz,
            weight=50.0,
        ))

    # Axis symmetry or rigid bore: u_r = 0 at r = r_min
    if fix_r_inner:
        conditions.append(DirichletBC(
            name="symmetry_ur",
            fields=("u_r",),
            selector_type="callable",
            selector=lambda X, ctx, _rmin=float(r_min): np.abs(X[:, 0] - _rmin) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ))

    # Internal pressure at r = r_min
    if abs(p_inner) > 0:
        _pi = float(p_inner)
        conditions.append(NeumannBC(
            name="inner_pressure",
            fields=("u_r",),
            selector_type="callable",
            selector=lambda X, ctx, _rmin=float(r_min): np.abs(X[:, 0] - _rmin) < 1e-6,
            value_fn=lambda X, ctx, _p=_pi: np.full((X.shape[0], 1), -_p, dtype=np.float32),
            weight=10.0,
        ))

    # Axial traction at z = z_max
    if abs(traction_z_top) > 0:
        _tz = float(traction_z_top)
        conditions.append(NeumannBC(
            name="axial_traction_top",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _zmax=float(z_max): np.abs(X[:, 1] - _zmax) < 1e-6,
            value_fn=lambda X, ctx, _t=_tz: np.full((X.shape[0], 1), _t, dtype=np.float32),
            weight=10.0,
        ))

    r_scale = float(r_max - r_min)
    z_scale = float(z_max - z_min)
    u_char = max(abs(p_inner), abs(traction_z_top), 1e6) / float(E) * float(z_max)

    return ProblemSpec(
        name="axisymmetric_linear_elasticity_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=tuple(conditions),
        sample_defaults={
            "n_col": 8000,
            "n_bc": 2000,
            "n_ic": 0,
        },
        scales=ScaleSpec(L=max(r_scale, z_scale), U=u_char),
        field_ranges={"u_r": (-u_char, u_char), "u_z": (-u_char, u_char)},
        references=(
            "Axisymmetric linear elasticity — cylindrical (r,z) coordinates.",
            "Timoshenko & Goodier, Theory of Elasticity, 3rd ed.",
        ),
        domain_bounds={"r": (float(r_min), float(r_max)), "z": (float(z_min), float(z_max))},
        solver_spec={"name": "fenics", "method": "axisymmetric_linear_elasticity",
                     "params": {"E": float(E), "nu": float(nu)}},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Thick-walled cylinder under internal pressure (Lamé — analytical)
# ══════════════════════════════════════════════════════════════════════════════

@register_preset("thick_walled_cylinder_lame")
def thick_walled_cylinder_lame_default(
    a: float = 20.0,         # mm — inner radius
    b: float = 60.0,         # mm — outer radius
    L: float = 100.0,        # mm — cylinder length
    p_a: float = 100e6,      # Pa — internal pressure
    p_b: float = 0.0,        # Pa — external pressure
    E: float = 2.1e11,
    nu: float = 0.3,
) -> ProblemSpec:
    """
    Thick-walled hollow cylinder under internal/external pressure (Lamé problem).

    Analytical solution (Lamé equations):
      σ_rr(r) = (a²p_a - b²p_b)/(b²-a²)  +  (p_b-p_a)·a²b²/[(b²-a²)·r²]
      σ_θθ(r) = (a²p_a - b²p_b)/(b²-a²)  -  (p_b-p_a)·a²b²/[(b²-a²)·r²]
      u_r(r)  = r/(E(b²-a²)) · [(1-ν)(a²p_a-b²p_b) + (1+ν)a²b²(p_a-p_b)/r²]

    Use ``lame_analytical(r, a, b, p_a, p_b, E, nu)`` to evaluate the
    ground truth at any radial position.

    Applications
    ------------
    Pressure vessels, gun barrels, hydraulic cylinders, thick rings.
    Benchmark problem for validating elasticity PINNs.
    """
    coords: CoordNames = ("r", "z")
    fields = ("u_r", "u_z")
    lam, mu = _lame(float(E), float(nu))

    pde = PDETermSpec(
        kind="axisymmetric_linear_elasticity",
        fields=fields,
        coords=coords,
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={
            "analytical": "lame_thick_walled_cylinder",
            "a": float(a), "b": float(b),
            "p_a": float(p_a), "p_b": float(p_b),
        },
    )

    _a, _b, _pa, _pb = float(a), float(b), float(p_a), float(p_b)
    _E, _nu = float(E), float(nu)

    # Radial displacement (analytical) at inner face: u_r(a)
    u_r_inner = _a / (_E * (_b**2 - _a**2)) * (
        (1 - _nu) * (_a**2 * _pa - _b**2 * _pb)
        + (1 + _nu) * _a**2 * _b**2 * (_pa - _pb) / _a**2
    )
    # Internal pressure as Neumann BC: σ_rr(r=a) = −p_a  (outward normal is −r̂)
    conditions = [
        NeumannBC(
            name="inner_pressure",
            fields=("u_r",),
            selector_type="callable",
            selector=lambda X, ctx, _r=_a: np.abs(X[:, 0] - _r) < 1e-6,
            value_fn=lambda X, ctx, _p=_pa: np.full((X.shape[0], 1), -_p, dtype=np.float32),
            weight=10.0,
        ),
        NeumannBC(
            name="outer_surface",
            fields=("u_r",),
            selector_type="callable",
            selector=lambda X, ctx, _r=_b: np.abs(X[:, 0] - _r) < 1e-6,
            value_fn=lambda X, ctx, _p=_pb: np.full((X.shape[0], 1), _p, dtype=np.float32),
            weight=10.0,
        ),
        DirichletBC(
            name="fixed_base_uz",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx: np.abs(X[:, 1]) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ),
    ]

    u_char = abs(u_r_inner) * 2 + 1e-10

    return ProblemSpec(
        name="thick_walled_cylinder_lame",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=tuple(conditions),
        sample_defaults={"n_col": 6000, "n_bc": 1500},
        scales=ScaleSpec(L=float(b), U=u_char),
        field_ranges={"u_r": (-u_char, u_char), "u_z": (-u_char, u_char)},
        references=(
            "Lamé (1852) — thick-walled cylinder under pressure.",
            "Timoshenko & Goodier, Theory of Elasticity §107.",
        ),
        domain_bounds={"r": (_a, _b), "z": (0.0, float(L))},
        solver_spec={"name": "analytical", "method": "lame_thick_walled_cylinder"},
    )


def lame_analytical(
    r: np.ndarray,
    a: float, b: float,
    p_a: float, p_b: float,
    E: float, nu: float,
) -> dict:
    """
    Lamé analytical solution for thick-walled cylinder.

    Parameters
    ----------
    r : array — radial coordinates  (a ≤ r ≤ b)

    Returns dict with keys: sigma_rr, sigma_tt, u_r
    """
    r = np.asarray(r, dtype=np.float64)
    denom = b**2 - a**2
    A = (a**2 * p_a - b**2 * p_b) / denom
    B = (p_a - p_b) * a**2 * b**2 / denom

    sigma_rr = A - B / r**2
    sigma_tt = A + B / r**2
    u_r = r / E * ((1 - nu) * A + (1 + nu) * B / r**2)

    return {"sigma_rr": sigma_rr, "sigma_tt": sigma_tt, "u_r": u_r}


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Threaded coupling (TC50)
# ══════════════════════════════════════════════════════════════════════════════

@register_preset("threaded_coupling_tc50_box")
def threaded_coupling_tc50_box_default(
    clearance: float = 0.1,        # mm — thread clearance
    thread_height: float = 0.8,    # mm
    offset: float = 1.0,           # mm — axial make-up offset
    E: float = 2.1e11,             # Pa — AISI 4145H steel
    nu: float = 0.3,
    traction_top: float = 1e8,     # Pa — axial traction at top shoulder
) -> ProblemSpec:
    """
    2D axisymmetric FEM preset for TC50 threaded coupling BOX body.

    Geometry (bounding box, actual profile is complex):
      r ∈ [r_bore, r_outer] = [50.0, 84.15] mm
      z ∈ [0, L_thread + L_shoulder] = [0, 140] mm

    The complex threaded profile is handled by the FEM mesh (mesh.msh).
    PINNeAPPle uses the bounding box for collocation sampling; actual geometry
    constraints come from FEM data or the mesh bridge.

    Boundary conditions:
      u_z = 0 at z = 0       — fixed nose (thread start)
      σ_zz = traction_top    — axial assembly load at top shoulder
      σ_rr free on profile   — thread contact forces (via FEM data)

    Loading the FEM solution
    ------------------------
    Use ``SolidMechanicsPipeline.from_fem_solution(solution_npy, spec)``
    to add the FEM data as DataConstraint automatically.
    """
    r_bore  = 50.0    # mm — bore radius (inner surface)
    r_outer = 84.15   # mm — outer radius of BOX
    z_min   = 0.0
    z_max   = 140.0   # mm — L_thread + L_shoulder

    lam, mu = _lame(float(E), float(nu))
    u_char = float(traction_top) / float(E) * z_max * 2

    pde = PDETermSpec(
        kind="axisymmetric_linear_elasticity",
        fields=("u_r", "u_z"),
        coords=("r", "z"),
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={
            "connection": "TC50",
            "body": "BOX",
            "clearance": float(clearance),
            "thread_height": float(thread_height),
            "pitch_mm": 6.35,
            "taper": 1/16,
            "standard": "V-0.038R thread form",
        },
    )

    conditions = [
        DirichletBC(
            name="fixed_nose_uz",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx: np.abs(X[:, 1] - z_min) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ),
        NeumannBC(
            name="traction_shoulder",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_max: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx, _t=float(traction_top): np.full(
                (X.shape[0], 1), _t, dtype=np.float32),
            weight=10.0,
        ),
    ]

    return ProblemSpec(
        name="threaded_coupling_tc50_box",
        dim=2,
        coords=("r", "z"),
        fields=("u_r", "u_z"),
        pde=pde,
        conditions=tuple(conditions),
        sample_defaults={"n_col": 8000, "n_bc": 2000},
        scales=ScaleSpec(L=r_outer, U=u_char),
        field_ranges={"u_r": (-u_char, u_char), "u_z": (-u_char, u_char)},
        references=(
            "TC50 rotary threaded coupling — BOX body geometry.",
            "Timoshenko & Goodier, Theory of Elasticity.",
        ),
        domain_bounds={"r": (r_bore, r_outer), "z": (z_min, z_max)},
        solver_spec={
            "name": "fenics",
            "method": "axisymmetric_linear_elasticity",
            "params": {"E": float(E), "nu": float(nu),
                       "mesh_file": "mesh.msh",
                       "clearance": float(clearance),
                       "thread_height": float(thread_height)},
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4 — TC50 BOX+PIN under combined loading (pressure + axial + torque)
# ══════════════════════════════════════════════════════════════════════════════

@register_preset("threaded_coupling_tc50_rotating")
def threaded_coupling_tc50_rotating_default(
    body: str = "BOX",          # "BOX" or "PIN"
    E: float = 2.1e11,          # Pa — AISI 4145H steel
    nu: float = 0.3,
    p_inner: float = 20e6,      # Pa — internal fluid pressure
    F_axial: float = 500e3,     # N  — axial load (compression positive)
    T_torque: float = 40e3,     # N·m — assembly + rotary torque
    rpm: float = 90.0,          # RPM (quasi-static — inertia neglected for now)
) -> ProblemSpec:
    """
    TC50 threaded coupling under combined loading.

    Fields  : u_r, u_z, u_θ  (radial, axial, hoop/torsional displacements)
    PDE     : axisymmetric linear elasticity (u_r, u_z) + torsion (u_θ)

    In linear elasticity the torsion problem decouples from the meridional
    problem. The torsional Navier equation is:

        ∂²u_θ/∂r² + (1/r)∂u_θ/∂r − u_θ/r² + ∂²u_θ/∂z² = 0

    Torsional stresses (contribute to full Von Mises):

        τ_rθ = μ (∂u_θ/∂r − u_θ/r)
        τ_θz = μ  ∂u_θ/∂z

    Full 6-component Von Mises:

        σ_vm = √(½ [(σ_rr−σ_zz)²+(σ_zz−σ_θθ)²+(σ_θθ−σ_rr)²
                    + 6(τ_rz²+τ_rθ²+τ_θz²)])

    Boundary conditions:
        u_z(r, z=0)  = 0          — fixed nose (thread start)
        u_θ(r, z=0)  = 0          — no rotation at fixed end
        σ_zz(r, z=L) = F/A_cross  — axial traction at top shoulder
        u_θ(r, z=L)  = θ_max · r  — prescribed make-up + rotary twist

    Loading
    -------
        rpm = 0   → static make-up only (pure torque T_torque)
        rpm > 0   → quasi-static rotation (T_torque includes rotary drag)
        p_inner   → bore pressure (σ_rr = −p at r = r_bore)
        F_axial   → axial tension/compression
    """
    body = str(body).upper()
    if body == "BOX":
        r_bore, r_outer, z_max = 50.0, 84.15, 140.0
    else:   # PIN
        r_bore, r_outer, z_max = 48.0, 62.0, 140.0
    z_min = 0.0

    lam, mu = _lame(float(E), float(nu))
    G = float(E) / (2 * (1 + float(nu)))

    # Cross-section area (annulus)
    A_cross = np.pi * (r_outer**2 - r_bore**2) * 1e-6  # m² (radii in mm)
    sigma_axial = float(F_axial) / max(A_cross, 1e-10)  # Pa

    # Characteristic twist angle at z=z_max  (θ_max in radians)
    # Using thin-ring approximation: θ ≈ T·L / (G·J)  where J = π(b⁴−a⁴)/2
    J = np.pi * ((r_outer * 1e-3) ** 4 - (r_bore * 1e-3) ** 4) / 2.0  # m⁴
    theta_max = float(T_torque) * (z_max * 1e-3) / max(G * J, 1e-30)  # rad
    # Prescribed hoop displacement at top: u_θ(r, z=L) = θ_max · r  [mm]
    uth_char = theta_max * r_outer  # mm — characteristic u_θ at outer edge

    u_char = max(
        abs(sigma_axial) / float(E) * z_max,
        abs(float(p_inner)) / float(E) * (r_outer - r_bore),
        abs(uth_char),
        1e-6,
    )

    pde = PDETermSpec(
        kind="axisymmetric_linear_elasticity_torsion",
        fields=("u_r", "u_z", "u_θ"),
        coords=("r", "z"),
        params={
            "E": float(E), "nu": float(nu),
            "lambda": lam, "mu": mu, "G": G,
            "p_inner": float(p_inner),
            "F_axial": float(F_axial),
            "T_torque": float(T_torque),
            "rpm": float(rpm),
        },
        meta={
            "connection": "TC50",
            "body": body,
            "formulation": "axisymmetric_rz + decoupled_torsion",
            "torsion_eqn": "d²u_θ/dr² + (1/r)du_θ/dr - u_θ/r² + d²u_θ/dz² = 0",
            "pitch_mm": 6.35,
            "taper": 1 / 16,
            "standard": "V-0.038R thread form",
            "theta_max_rad": float(theta_max),
        },
    )

    _p  = float(p_inner)
    _sa = float(sigma_axial)
    _th = float(theta_max)

    conditions = [
        # Fixed nose: u_z = 0 at z = 0
        DirichletBC(
            name="fixed_nose_uz",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_min: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ),
        # No rotation at fixed end: u_θ = 0 at z = 0
        DirichletBC(
            name="fixed_nose_uth",
            fields=("u_θ",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_min: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ),
        # Axial traction at top shoulder: σ_zz = F/A
        NeumannBC(
            name="axial_traction_top",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_max: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx, _s=_sa: np.full((X.shape[0], 1), _s, dtype=np.float32),
            weight=10.0,
        ),
        # Internal bore pressure: σ_rr = −p at r = r_bore
        NeumannBC(
            name="bore_pressure",
            fields=("u_r",),
            selector_type="callable",
            selector=lambda X, ctx, _r=r_bore: np.abs(X[:, 0] - _r) < 1e-6,
            value_fn=lambda X, ctx, _p2=_p: np.full((X.shape[0], 1), -_p2, dtype=np.float32),
            weight=10.0,
        ),
        # Prescribed twist at top: u_θ(r, z=L) = θ_max · r
        DirichletBC(
            name="prescribed_twist_top",
            fields=("u_θ",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_max: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx, _t=_th: (_t * X[:, 0:1]).astype(np.float32),
            weight=50.0,
        ),
    ]

    return ProblemSpec(
        name=f"threaded_coupling_tc50_rotating_{body.lower()}",
        dim=2,
        coords=("r", "z"),
        fields=("u_r", "u_z", "u_θ"),
        pde=pde,
        conditions=tuple(conditions),
        sample_defaults={"n_col": 10000, "n_bc": 2500},
        scales=ScaleSpec(L=r_outer, U=u_char),
        field_ranges={
            "u_r": (-u_char, u_char),
            "u_z": (-u_char, u_char),
            "u_θ": (-uth_char * 1.2, uth_char * 1.2),
        },
        references=(
            "TC50 rotary threaded coupling — combined loading.",
            "Timoshenko & Goodier, Theory of Elasticity, ch. 11 (torsion).",
        ),
        domain_bounds={"r": (r_bore, r_outer), "z": (z_min, z_max)},
        solver_spec={
            "name": "fenics",
            "method": "axisymmetric_linear_elasticity_torsion",
            "params": {
                "E": float(E), "nu": float(nu),
                "p_inner": float(p_inner),
                "F_axial": float(F_axial),
                "T_torque": float(T_torque),
                "mesh_file": "mesh_tc50.msh",
            },
        },
    )


@register_preset("threaded_coupling_tc50_pin")
def threaded_coupling_tc50_pin_default(
    thread_height: float = 0.8,
    E: float = 2.1e11,
    nu: float = 0.3,
    traction_top: float = 1e8,
) -> ProblemSpec:
    """TC50 threaded coupling PIN body — same physics as BOX, different geometry bounds."""
    pin_bore_r = 48.0   # mm — ID bore radius
    r_outer    = 62.0   # mm — approximate outer radius at shoulder
    z_max      = 140.0
    lam, mu    = _lame(float(E), float(nu))
    u_char     = float(traction_top) / float(E) * z_max * 2

    pde = PDETermSpec(
        kind="axisymmetric_linear_elasticity",
        fields=("u_r", "u_z"),
        coords=("r", "z"),
        params={"E": float(E), "nu": float(nu), "lambda": lam, "mu": mu},
        meta={"connection": "TC50", "body": "PIN",
              "thread_height": float(thread_height)},
    )

    conditions = [
        DirichletBC(
            name="fixed_nose_uz",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx: np.abs(X[:, 1]) < 1e-6,
            value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
            weight=50.0,
        ),
        NeumannBC(
            name="traction_top",
            fields=("u_z",),
            selector_type="callable",
            selector=lambda X, ctx, _z=z_max: np.abs(X[:, 1] - _z) < 1e-6,
            value_fn=lambda X, ctx, _t=float(traction_top): np.full(
                (X.shape[0], 1), _t, dtype=np.float32),
            weight=10.0,
        ),
    ]

    return ProblemSpec(
        name="threaded_coupling_tc50_pin",
        dim=2, coords=("r", "z"), fields=("u_r", "u_z"),
        pde=pde, conditions=tuple(conditions),
        sample_defaults={"n_col": 8000, "n_bc": 2000},
        scales=ScaleSpec(L=r_outer, U=u_char),
        field_ranges={"u_r": (-u_char, u_char), "u_z": (-u_char, u_char)},
        references=("TC50 threaded coupling — PIN body.",),
        domain_bounds={"r": (pin_bore_r, r_outer), "z": (0.0, z_max)},
        solver_spec={"name": "fenics", "method": "axisymmetric_linear_elasticity",
                     "params": {"E": float(E), "nu": float(nu)}},
    )
