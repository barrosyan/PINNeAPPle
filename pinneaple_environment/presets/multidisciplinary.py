"""Multidisciplinary problem presets.

Covers cross-domain PDE and ODE problems spanning climate science, materials
discovery, finance, pharmacokinetics, and social systems.

Each preset returns a ``ProblemSpec`` ready for use with the pinneaple
training pipeline, data generation, and digital twin modules.

Domains
-------
Climate Modeling
  - climate_atmosphere_2d   : 2D shallow water equations (atmospheric dynamics)
  - climate_ocean_gyre      : Ocean gyre circulation (Stommel beta-plane model)

Material Discovery
  - crystal_phonon          : 1D phonon Boltzmann transport (lattice temperature)
  - material_fracture_2d    : Phase-field fracture model 2D (Cahn-Hilliard + elasticity)

Financial Modeling
  - black_scholes_1d        : Black-Scholes PDE for option pricing (1D in S)
  - heston_pde_2d           : Heston stochastic volatility PDE (2D in S, v)

Pharmacokinetics
  - pk_two_compartment      : Two-compartment PK ODE system
  - drug_diffusion_tissue   : Drug diffusion and metabolism in 2D tissue

Social Systems
  - sir_epidemic            : SIR epidemic ODE system
  - opinion_dynamics_2d     : 2D spatial opinion dynamics PDE
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ..spec import PDETermSpec, ProblemSpec
from ..conditions import DirichletBC, NeumannBC, InitialCondition
from ..scales import ScaleSpec
from ..typing import CoordNames
from .registry import register_preset


# ===========================================================================
# CLIMATE MODELING
# ===========================================================================

@register_preset("climate_atmosphere_2d")
def climate_atmosphere_2d(
    g: float = 9.81,
    f: float = 1e-4,
    H0: float = 1000.0,
    U_ref: float = 10.0,
    lon_min: float = 0.0,
    lon_max: float = 360.0,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
) -> ProblemSpec:
    """2D shallow water equations for atmospheric dynamics.

    PDE (non-rotating form with Coriolis source term):
        ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
        ∂u/∂t + u ∂u/∂x + v ∂u/∂y - f v + g ∂h/∂x = 0
        ∂v/∂t + u ∂v/∂x + v ∂v/∂y + f u + g ∂h/∂y = 0

    Fields: h (fluid height/geopotential), u (east-west velocity), v (north-south velocity).

    Parameters
    ----------
    g      : gravitational acceleration (m/s²)
    f      : Coriolis parameter (s⁻¹); mid-latitude approximation
    H0     : reference layer thickness (m)
    U_ref  : reference wind speed (m/s)
    lon_min/lon_max : longitude bounds (degrees)
    lat_min/lat_max : latitude bounds (degrees)
    """
    coords: CoordNames = ("x", "y", "t")
    fields = ("h", "u", "v")

    pde = PDETermSpec(
        kind="shallow_water_2d",
        fields=fields,
        coords=coords,
        params={"g": g, "f": f, "H0": H0},
        meta={
            "note": (
                "Linearised rotating shallow-water equations on a beta-plane. "
                "x = longitude (deg), y = latitude (deg), t = time (s)."
            ),
            "coriolis_term": "f*v, -f*u",
            "gravity_wave_speed_estimate": math.sqrt(g * H0),
        },
    )

    ic_h = InitialCondition(
        name="ic_h",
        fields=("h",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), H0, dtype=np.float32),
        weight=10.0,
    )

    ic_uv = InitialCondition(
        name="ic_uv",
        fields=("u", "v"),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=10.0,
    )

    # Periodic boundary in longitude; solid-wall Neumann at poles
    bc_pole_south = NeumannBC(
        name="bc_pole_south",
        fields=("u", "v"),
        selector_type="tag",
        selector={"tag": "pole_south"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=5.0,
    )

    bc_pole_north = NeumannBC(
        name="bc_pole_north",
        fields=("u", "v"),
        selector_type="tag",
        selector={"tag": "pole_north"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=5.0,
    )

    L_ref = (lon_max - lon_min) * 111_000.0  # degrees -> metres (approx)

    return ProblemSpec(
        name="climate_atmosphere_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_h, ic_uv, bc_pole_south, bc_pole_north),
        sample_defaults={"n_col": 200_000, "n_ic": 20_000, "n_bc": 10_000},
        scales=ScaleSpec(L=L_ref, U=U_ref),
        field_ranges={"h": (0.5 * H0, 1.5 * H0), "u": (-U_ref, U_ref), "v": (-U_ref, U_ref)},
        references=(
            "Vallis, G.K. (2006). Atmospheric and Oceanic Fluid Dynamics. Cambridge University Press.",
        ),
        domain_bounds={"x": (lon_min, lon_max), "y": (lat_min, lat_max), "t": (0.0, 86400.0)},
        solver_spec={
            "name": "spectral",
            "method": "shallow_water_leapfrog",
            "params": {"nx": 360, "ny": 180, "nt": 1440, "dt": 60.0},
        },
    )


@register_preset("climate_ocean_gyre")
def climate_ocean_gyre(
    beta: float = 2e-11,
    tau0: float = 0.1,
    r: float = 1e-7,
    rho0: float = 1025.0,
    H: float = 500.0,
    L: float = 1e6,
    W: float = 5e5,
) -> ProblemSpec:
    """Ocean gyre circulation — Stommel beta-plane model.

    Stommel model PDE (for barotropic streamfunction psi):
        r ∇²ψ + β ∂ψ/∂x = (1/ρ₀ H) curl(τ)

    where curl(τ) is the wind-stress curl (prescribed forcing).
    Fields: psi (barotropic streamfunction, m²/s).

    Parameters
    ----------
    beta  : planetary vorticity gradient β = df/dy (m⁻¹ s⁻¹)
    tau0  : wind stress amplitude (N/m²)
    r     : Rayleigh bottom friction coefficient (s⁻¹)
    rho0  : reference ocean density (kg/m³)
    H     : layer depth (m)
    L     : zonal basin width (m)
    W     : meridional basin width (m)
    """
    coords: CoordNames = ("x", "y")
    fields = ("psi",)

    pde = PDETermSpec(
        kind="stommel_gyre_2d",
        fields=fields,
        coords=coords,
        params={
            "beta": beta,
            "tau0": tau0,
            "r": r,
            "rho0": rho0,
            "H": H,
        },
        meta={
            "note": (
                "Stommel (1948) wind-driven gyre. "
                "Wind stress curl forcing: (tau0 * pi / W) * sin(pi * y / W). "
                "Western boundary current (Stommel solution) arises from beta term."
            ),
            "forcing": "curl_tau = (tau0 * pi / W) * sin(pi * y / W)",
        },
    )

    # No-slip streamfunction on all basin walls (psi = 0 on boundary)
    bc_walls = DirichletBC(
        name="bc_walls",
        fields=("psi",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    U_ref = tau0 / (rho0 * r * H)

    return ProblemSpec(
        name="climate_ocean_gyre",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc_walls,),
        sample_defaults={"n_col": 80_000, "n_bc": 10_000},
        scales=ScaleSpec(L=L, U=U_ref),
        field_ranges={"psi": (-U_ref * L, U_ref * L)},
        references=(
            "Stommel, H. (1948). The westward intensification of wind-driven ocean currents. "
            "Trans. Am. Geophys. Union, 29(2), 202-206.",
        ),
        domain_bounds={"x": (0.0, L), "y": (0.0, W)},
        solver_spec={
            "name": "fdm",
            "method": "stommel_sor",
            "params": {"nx": 200, "ny": 100, "tol": 1e-8, "omega": 1.8},
        },
    )


# ===========================================================================
# MATERIAL DISCOVERY
# ===========================================================================

@register_preset("crystal_phonon")
def crystal_phonon(
    k: float = 150.0,
    tau: float = 1e-12,
    vg: float = 3000.0,
    T_hot: float = 320.0,
    T_cold: float = 280.0,
    L_domain: float = 1e-6,
) -> ProblemSpec:
    """Phonon transport / lattice dynamics in 1D — simplified Boltzmann transport.

    Simplified phonon BTE (gray approximation):
        ∂T/∂t + vg ∂T/∂x = -(T - T_eq) / tau + k/Cv ∂²T/∂x²

    where T is the lattice temperature and the equation reduces in steady-state
    to a diffusion-relaxation balance that captures ballistic-to-diffusive
    crossover in nanoscale crystals.

    Fields: T (lattice temperature, K).

    Parameters
    ----------
    k   : effective thermal conductivity (W/m·K)
    tau : phonon relaxation time (s)
    vg  : phonon group velocity (m/s)
    T_hot/T_cold : boundary temperatures (K)
    L_domain : sample length (m)
    """
    coords: CoordNames = ("x", "t")
    fields = ("T",)

    Kn = vg * tau / L_domain   # Knudsen number

    pde = PDETermSpec(
        kind="phonon_bte_1d_gray",
        fields=fields,
        coords=coords,
        params={"k": k, "tau": tau, "vg": vg, "Kn": Kn},
        meta={
            "note": (
                "Gray-medium phonon BTE (Callaway model). "
                "Kn = vg * tau / L: diffusive limit Kn << 1, ballistic limit Kn >> 1."
            ),
            "Knudsen_number": Kn,
            "fourier_length_scale_nm": L_domain * 1e9,
        },
    )

    ic_T = InitialCondition(
        name="ic_T",
        fields=("T",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 1], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), 0.5 * (T_hot + T_cold), dtype=np.float32),
        weight=10.0,
    )

    bc_hot = DirichletBC(
        name="bc_hot",
        fields=("T",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), T_hot, dtype=np.float32),
        weight=20.0,
    )

    bc_cold = DirichletBC(
        name="bc_cold",
        fields=("T",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], L_domain),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), T_cold, dtype=np.float32),
        weight=20.0,
    )

    t_end = 10.0 * tau

    return ProblemSpec(
        name="crystal_phonon",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_T, bc_hot, bc_cold),
        sample_defaults={"n_col": 50_000, "n_ic": 5_000, "n_bc": 5_000},
        scales=ScaleSpec(L=L_domain, U=vg, alpha=k),
        field_ranges={"T": (T_cold, T_hot)},
        references=(
            "Majumdar, A. (1993). Microscale heat conduction in dielectric thin films. "
            "J. Heat Transfer, 115(1), 7-16.",
        ),
        domain_bounds={"x": (0.0, L_domain), "t": (0.0, t_end)},
        solver_spec={
            "name": "fdm",
            "method": "phonon_bte_1d",
            "params": {"nx": 512, "nt": 1000, "dt": t_end / 1000},
        },
    )


@register_preset("material_fracture_2d")
def material_fracture_2d(
    E: float = 210e9,
    nu: float = 0.3,
    Gc: float = 2700.0,
    l0: float = 5e-3,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> ProblemSpec:
    """Phase-field fracture model 2D — Bourdin-Francfort-Marigo formulation.

    Coupled PDEs:
        Elasticity:  div[ (1-phi)² C : eps(u) ] = 0
        Phase-field: Gc/l0 * phi - Gc l0 ∇²phi = 2(1-phi) H

    where phi in [0,1] is the damage field, H is the crack driving force
    (max strain energy history), and C is the elastic stiffness.

    Fields: ux (x-displacement), uy (y-displacement), phi (damage field).

    Parameters
    ----------
    E   : Young's modulus (Pa)
    nu  : Poisson's ratio
    Gc  : critical energy release rate / fracture energy (J/m²)
    l0  : phase-field regularization length (m)
    Lx/Ly : domain dimensions (m)
    """
    coords: CoordNames = ("x", "y")
    fields = ("ux", "uy", "phi")

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    pde = PDETermSpec(
        kind="phase_field_fracture_2d",
        fields=fields,
        coords=coords,
        params={
            "E": E, "nu": nu, "lambda": lam, "mu": mu,
            "Gc": Gc, "l0": l0,
        },
        meta={
            "note": (
                "AT-2 phase-field fracture (Bourdin et al. 2000). "
                "Coupled staggered solve: elasticity then phase-field update. "
                "phi=0 undamaged, phi=1 fully fractured."
            ),
            "regularization_ratio": l0 / min(Lx, Ly),
        },
    )

    # Fixed bottom, prescribed top displacement, free sides
    bc_bottom = DirichletBC(
        name="bc_bottom_fixed",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "bottom"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=50.0,
    )

    bc_top = DirichletBC(
        name="bc_top_displacement",
        fields=("uy",),
        selector_type="tag",
        selector={"tag": "top"},
        value_fn=lambda X, ctx: np.full(
            (X.shape[0], 1),
            ctx.get("u_applied", 1e-4),
            dtype=np.float32,
        ),
        weight=50.0,
    )

    bc_sides = NeumannBC(
        name="bc_sides_free",
        fields=("ux", "uy"),
        selector_type="tag",
        selector={"tag": "sides"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float32),
        weight=5.0,
    )

    # Neumann (zero flux) for phase-field on all boundaries
    bc_phi = NeumannBC(
        name="bc_phi_zero_flux",
        fields=("phi",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="material_fracture_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc_bottom, bc_top, bc_sides, bc_phi),
        sample_defaults={"n_col": 100_000, "n_bc": 20_000},
        scales=ScaleSpec(L=Lx, U=1.0, alpha=E),
        field_ranges={"ux": (-1e-3, 1e-3), "uy": (-1e-3, 1e-3), "phi": (0.0, 1.0)},
        references=(
            "Bourdin, B., Francfort, G.A., Marigo, J.-J. (2000). "
            "Numerical experiments in revisited brittle fracture. "
            "J. Mech. Phys. Solids, 48(4), 797-826.",
        ),
        domain_bounds={"x": (0.0, Lx), "y": (0.0, Ly)},
        solver_spec={
            "name": "fenics",
            "formulation": "phase_field_fracture_staggered",
            "params": {"mesh_size": l0 / 2.0, "degree": 1, "max_iter": 1000},
        },
    )


# ===========================================================================
# FINANCIAL MODELING
# ===========================================================================

@register_preset("black_scholes_1d")
def black_scholes_1d(
    sigma: float = 0.2,
    r: float = 0.05,
    K: float = 100.0,
    T: float = 1.0,
) -> ProblemSpec:
    """Black-Scholes PDE for European call option pricing.

    PDE (in forward-time τ = T - t convention):
        ∂V/∂τ = 0.5 σ² S² ∂²V/∂S² + r S ∂V/∂S - r V

    Terminal condition (τ=0, i.e. t=T):  V(S, T) = max(S - K, 0)
    Boundary conditions:
        V(0, t)  = 0  (worthless at S=0)
        V(S→∞)  ~ S - K e^{-r(T-t)}  (linear for large S, Neumann approx)

    Fields: V (option value).

    Parameters
    ----------
    sigma : volatility (annualised, dimensionless)
    r     : risk-free interest rate (annualised)
    K     : strike price
    T     : time to maturity (years)
    """
    coords: CoordNames = ("S", "tau")   # tau = T - t (time-to-expiry)
    fields = ("V",)

    S_max = 3.0 * K

    pde = PDETermSpec(
        kind="black_scholes_1d",
        fields=fields,
        coords=coords,
        params={"sigma": sigma, "r": r, "K": K},
        meta={
            "note": (
                "Black-Scholes PDE in (S, tau) where tau = T - t. "
                "Terminal condition at tau=0 is the European call payoff max(S-K, 0). "
                "d1 = (log(S/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau)), "
                "Exact BS price available for validation."
            ),
            "analytical_solution": "BS_call_formula",
            "d1_formula": "(log(S/K) + (r + 0.5*sigma^2)*tau) / (sigma*sqrt(tau))",
        },
    )

    # Initial condition (terminal payoff at tau=0)
    ic_payoff = InitialCondition(
        name="ic_terminal_payoff",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 1], 0.0),
        value_fn=lambda X, ctx: np.maximum(X[:, 0:1] - K, 0.0).astype(np.float32),
        weight=20.0,
    )

    # V = 0 at S = 0
    bc_S0 = DirichletBC(
        name="bc_S_zero",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    # ∂V/∂S = 1 at S = S_max (delta -> 1 for deep in-the-money)
    bc_Smax = NeumannBC(
        name="bc_S_max",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], S_max),
        value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="black_scholes_1d",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_payoff, bc_S0, bc_Smax),
        sample_defaults={"n_col": 80_000, "n_ic": 10_000, "n_bc": 5_000},
        scales=ScaleSpec(L=K, U=K),
        field_ranges={"V": (0.0, S_max)},
        references=(
            "Black, F., Scholes, M. (1973). The pricing of options and corporate liabilities. "
            "J. Political Economy, 81(3), 637-654.",
        ),
        domain_bounds={"S": (0.0, S_max), "tau": (0.0, T)},
        solver_spec={
            "name": "fdm",
            "method": "crank_nicolson_bs",
            "params": {"nS": 500, "ntau": 200, "S_max": S_max},
        },
    )


@register_preset("heston_pde_2d")
def heston_pde_2d(
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float = -0.7,
    r: float = 0.05,
    K: float = 100.0,
    T: float = 1.0,
    S_max: float = 300.0,
    v_max: float = 1.0,
) -> ProblemSpec:
    """Heston stochastic volatility PDE for European option pricing (2D).

    PDE (in forward-time τ = T - t):
        ∂V/∂τ = 0.5 S² v ∂²V/∂S²
              + ρ σ_v S v ∂²V/∂S∂v
              + 0.5 σ_v² v ∂²V/∂v²
              + r S ∂V/∂S
              + κ(θ - v) ∂V/∂v
              - r V

    Fields: V (option value, function of S, v, tau).

    Parameters
    ----------
    kappa   : mean reversion speed of variance
    theta   : long-run average variance
    sigma_v : vol-of-vol
    rho     : correlation between asset and variance Brownian motions
    r       : risk-free rate
    K       : strike price
    T       : maturity (years)
    S_max   : maximum asset price in domain
    v_max   : maximum variance in domain
    """
    coords: CoordNames = ("S", "v", "tau")
    fields = ("V",)

    pde = PDETermSpec(
        kind="heston_pde_2d",
        fields=fields,
        coords=coords,
        params={
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v,
            "rho": rho, "r": r, "K": K,
        },
        meta={
            "note": (
                "Heston (1993) stochastic volatility model PDE. "
                "Feller condition (2 kappa theta >= sigma_v^2) for v>0: "
                f"{2 * kappa * theta:.4f} >= {sigma_v ** 2:.4f}: "
                f"{'satisfied' if 2 * kappa * theta >= sigma_v ** 2 else 'NOT satisfied'}."
            ),
            "feller_condition_satisfied": 2 * kappa * theta >= sigma_v ** 2,
        },
    )

    # Terminal payoff (European call)
    ic_payoff = InitialCondition(
        name="ic_terminal_payoff",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.maximum(X[:, 0:1] - K, 0.0).astype(np.float32),
        weight=20.0,
    )

    # V = 0 at S = 0
    bc_S0 = DirichletBC(
        name="bc_S_zero",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    # ∂V/∂S = 1 at S = S_max
    bc_Smax = NeumannBC(
        name="bc_S_max",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], S_max),
        value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    # ∂V/∂v = 0 at v = v_max (far-field in variance)
    bc_vmax = NeumannBC(
        name="bc_v_max",
        fields=("V",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 1], v_max),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="heston_pde_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_payoff, bc_S0, bc_Smax, bc_vmax),
        sample_defaults={"n_col": 200_000, "n_ic": 20_000, "n_bc": 10_000},
        scales=ScaleSpec(L=K, U=K),
        field_ranges={"V": (0.0, S_max)},
        references=(
            "Heston, S.L. (1993). A closed-form solution for options with stochastic volatility. "
            "Rev. Financial Studies, 6(2), 327-343.",
        ),
        domain_bounds={"S": (0.0, S_max), "v": (0.0, v_max), "tau": (0.0, T)},
        solver_spec={
            "name": "fdm",
            "method": "adi_heston",
            "params": {"nS": 200, "nv": 100, "ntau": 200, "S_max": S_max, "v_max": v_max},
        },
    )


# ===========================================================================
# PHARMACOKINETICS
# ===========================================================================

@register_preset("pk_two_compartment")
def pk_two_compartment(
    k12: float = 0.5,
    k21: float = 0.3,
    kel: float = 0.2,
    V1: float = 10.0,
    dose: float = 500.0,
    t_end: float = 24.0,
) -> ProblemSpec:
    """Two-compartment pharmacokinetic (PK) model — ODE system.

    ODE system:
        dC1/dt = -(k12 + kel) C1 + k21 C2 + dose_rate(t) / V1
        dC2/dt = k12 C1 - k21 C2

    where C1 is the drug concentration in the central compartment (blood/plasma)
    and C2 is the concentration in the peripheral compartment (tissue).
    IV bolus initial conditions: C1(0) = dose / V1, C2(0) = 0.

    Fields: C1 (central compartment concentration), C2 (peripheral).

    Parameters
    ----------
    k12  : transfer rate central -> peripheral (h⁻¹)
    k21  : transfer rate peripheral -> central (h⁻¹)
    kel  : elimination rate from central compartment (h⁻¹)
    V1   : volume of distribution of central compartment (L)
    dose : IV bolus dose (mg)
    t_end: simulation end time (h)
    """
    coords: CoordNames = ("t",)
    fields = ("C1", "C2")

    C1_0 = dose / V1

    pde = PDETermSpec(
        kind="pk_two_compartment_ode",
        fields=fields,
        coords=coords,
        params={"k12": k12, "k21": k21, "kel": kel, "V1": V1},
        meta={
            "note": (
                "Two-compartment PK model (IV bolus). "
                "C1 = central (plasma), C2 = peripheral (tissue). "
                "Half-life estimate (mono-exponential approx): "
                f"{math.log(2) / kel:.2f} h."
            ),
            "half_life_estimate_h": math.log(2) / kel,
            "C1_0_mg_per_L": C1_0,
            "AUC_estimate": C1_0 / kel,
        },
    )

    ic_C1 = InitialCondition(
        name="ic_C1",
        fields=("C1",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), C1_0, dtype=np.float32),
        weight=20.0,
    )

    ic_C2 = InitialCondition(
        name="ic_C2",
        fields=("C2",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="pk_two_compartment",
        dim=0,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_C1, ic_C2),
        sample_defaults={"n_col": 10_000, "n_ic": 500},
        scales=ScaleSpec(L=t_end, U=C1_0),
        field_ranges={"C1": (0.0, C1_0), "C2": (0.0, C1_0 * k12 / (k21 + kel))},
        references=(
            "Rowland, M., Tozer, T.N. (2011). Clinical Pharmacokinetics and Pharmacodynamics. "
            "Lippincott Williams & Wilkins, 4th ed.",
        ),
        domain_bounds={"t": (0.0, t_end)},
        solver_spec={
            "name": "scipy",
            "method": "solve_ivp",
            "params": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10},
        },
    )


@register_preset("drug_diffusion_tissue")
def drug_diffusion_tissue(
    D: float = 1e-10,
    lam: float = 5e-5,
    C0: float = 1.0,
    Lx: float = 1e-2,
    Ly: float = 5e-3,
    t_end: float = 3600.0,
) -> ProblemSpec:
    """Drug diffusion and first-order metabolism in 2D tissue.

    PDE (reaction-diffusion):
        ∂C/∂t = D (∂²C/∂x² + ∂²C/∂y²) - λ C

    where C is drug concentration, D is the effective diffusion coefficient
    in tissue, and λ is the first-order elimination rate.

    Fields: C (drug concentration, e.g. mg/mL or mol/m³).

    Parameters
    ----------
    D    : diffusion coefficient (m²/s), typical tissue: 1e-11 to 1e-9 m²/s
    lam  : first-order elimination / metabolism rate (s⁻¹)
    C0   : initial / inlet concentration (same units as C)
    Lx/Ly : tissue domain dimensions (m)
    t_end : simulation duration (s)
    """
    coords: CoordNames = ("x", "y", "t")
    fields = ("C",)

    diff_time = Lx ** 2 / D        # diffusion time scale (s)
    elim_time = 1.0 / lam           # elimination time scale (s)

    pde = PDETermSpec(
        kind="reaction_diffusion_2d",
        fields=fields,
        coords=coords,
        params={"D": D, "lambda": lam},
        meta={
            "note": (
                "Linear reaction-diffusion: dC/dt = D nabla^2 C - lambda C. "
                "Damkohler number Da = lambda * Lx^2 / D: "
                f"{lam * Lx ** 2 / D:.3f}."
            ),
            "Damkohler_number": lam * Lx ** 2 / D,
            "diffusion_time_s": diff_time,
            "elimination_time_s": elim_time,
        },
    )

    ic_C = InitialCondition(
        name="ic_C",
        fields=("C",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=10.0,
    )

    # Drug source at x=0 (capillary wall or application site)
    bc_source = DirichletBC(
        name="bc_source",
        fields=("C",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), C0, dtype=np.float32),
        weight=20.0,
    )

    # Zero-flux on remaining boundaries (insulated tissue)
    bc_no_flux = NeumannBC(
        name="bc_no_flux",
        fields=("C",),
        selector_type="tag",
        selector={"tag": "no_flux"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="drug_diffusion_tissue",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_C, bc_source, bc_no_flux),
        sample_defaults={"n_col": 100_000, "n_ic": 10_000, "n_bc": 10_000},
        scales=ScaleSpec(L=Lx, U=C0, alpha=D),
        field_ranges={"C": (0.0, C0)},
        references=(
            "Crank, J. (1975). The Mathematics of Diffusion. Oxford University Press, 2nd ed.",
        ),
        domain_bounds={"x": (0.0, Lx), "y": (0.0, Ly), "t": (0.0, t_end)},
        solver_spec={
            "name": "fdm",
            "method": "crank_nicolson_2d",
            "params": {"nx": 100, "ny": 50, "nt": 500},
        },
    )


# ===========================================================================
# SOCIAL SYSTEMS
# ===========================================================================

@register_preset("sir_epidemic")
def sir_epidemic(
    beta: float = 0.3,
    gamma: float = 0.1,
    N: float = 1e6,
    I0: float = 100.0,
    t_end: float = 200.0,
) -> ProblemSpec:
    """SIR epidemic model — ODE system.

    ODE system:
        dS/dt = -beta S I / N
        dI/dt =  beta S I / N - gamma I
        dR/dt =  gamma I

    with conservation S + I + R = N.

    Fields: S (susceptible), I (infected), R (recovered).

    Parameters
    ----------
    beta  : effective contact / transmission rate (day⁻¹)
    gamma : recovery rate (day⁻¹); mean infectious period = 1/gamma days
    N     : total population size
    I0    : initial number of infected individuals
    t_end : simulation end time (days)
    """
    coords: CoordNames = ("t",)
    fields = ("S", "I", "R")

    S0 = N - I0
    R0_number = beta / gamma   # basic reproduction number

    pde = PDETermSpec(
        kind="sir_ode",
        fields=fields,
        coords=coords,
        params={"beta": beta, "gamma": gamma, "N": N},
        meta={
            "note": (
                "Classic SIR compartmental model (Kermack-McKendrick 1927). "
                "Basic reproduction number R0 = beta/gamma = "
                f"{R0_number:.2f}. "
                "Epidemic occurs if R0 > 1."
            ),
            "R0": R0_number,
            "epidemic_threshold_exceeded": R0_number > 1.0,
            "herd_immunity_threshold": 1.0 - 1.0 / R0_number if R0_number > 1.0 else 0.0,
        },
    )

    ic_S = InitialCondition(
        name="ic_S",
        fields=("S",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), S0, dtype=np.float32),
        weight=20.0,
    )

    ic_I = InitialCondition(
        name="ic_I",
        fields=("I",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.full((X.shape[0], 1), I0, dtype=np.float32),
        weight=20.0,
    )

    ic_R = InitialCondition(
        name="ic_R",
        fields=("R",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 0], 0.0),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=20.0,
    )

    return ProblemSpec(
        name="sir_epidemic",
        dim=0,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_S, ic_I, ic_R),
        sample_defaults={"n_col": 10_000, "n_ic": 500},
        scales=ScaleSpec(L=t_end, U=N),
        field_ranges={"S": (0.0, N), "I": (0.0, N), "R": (0.0, N)},
        references=(
            "Kermack, W.O., McKendrick, A.G. (1927). "
            "A contribution to the mathematical theory of epidemics. "
            "Proc. Royal Soc. London A, 115, 700-721.",
        ),
        domain_bounds={"t": (0.0, t_end)},
        solver_spec={
            "name": "scipy",
            "method": "solve_ivp",
            "params": {"method": "RK45", "rtol": 1e-10, "atol": 1e-12},
        },
    )


@register_preset("opinion_dynamics_2d")
def opinion_dynamics_2d(
    D: float = 0.01,
    alpha: float = 1.0,
    Lx: float = 1.0,
    Ly: float = 1.0,
    t_end: float = 10.0,
) -> ProblemSpec:
    """2D spatial opinion dynamics PDE — continuum Hegselmann-Krause model.

    PDE (nonlinear reaction-diffusion on opinion field u ∈ [-1, 1]):
        ∂u/∂t = D ∇²u + α f(u)

    where f(u) = u(1 - u²) is a bistable reaction term (double-well potential
    derivative) that drives opinions toward the extremes ±1, while diffusion
    D ∇²u models spatial influence / social interaction across neighbours.

    Homogeneous Neumann BCs model a closed society (no opinion flux at boundary).

    Fields: u (spatial opinion field, u ∈ [-1, 1]).

    Parameters
    ----------
    D      : spatial diffusion coefficient (opinion spread, m²/time)
    alpha  : interaction / polarisation strength
    Lx/Ly  : spatial domain dimensions
    t_end  : simulation end time
    """
    coords: CoordNames = ("x", "y", "t")
    fields = ("u",)

    pde = PDETermSpec(
        kind="opinion_dynamics_2d",
        fields=fields,
        coords=coords,
        params={"D": D, "alpha": alpha},
        meta={
            "note": (
                "Continuum opinion dynamics: du/dt = D nabla^2 u + alpha * u*(1 - u^2). "
                "Bistable (Allen-Cahn) reaction term drives clustering to ±1. "
                "D controls spatial mixing; alpha controls polarisation rate. "
                "Diffusion length scale: sqrt(D/alpha) = "
                f"{math.sqrt(D / max(alpha, 1e-12)):.4f}."
            ),
            "diffusion_length_scale": math.sqrt(D / max(alpha, 1e-12)),
            "model_type": "Allen-Cahn / continuum Hegselmann-Krause",
            "reference_Hegselmann": "Hegselmann & Krause (2002). Opinion dynamics and bounded confidence.",
        },
    )

    # Random-ish initial condition centred near zero (via callable)
    ic_u = InitialCondition(
        name="ic_u_random",
        fields=("u",),
        selector_type="callable",
        selector=lambda X, ctx: np.isclose(X[:, 2], 0.0),
        value_fn=lambda X, ctx: (0.1 * np.sin(2.0 * math.pi * X[:, 0:1] / Lx)
                                  * np.cos(2.0 * math.pi * X[:, 1:2] / Ly)).astype(np.float32),
        weight=10.0,
    )

    # Zero-flux (Neumann) on all boundaries — closed society
    bc_no_flux = NeumannBC(
        name="bc_no_flux",
        fields=("u",),
        selector_type="tag",
        selector={"tag": "boundary"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=5.0,
    )

    return ProblemSpec(
        name="opinion_dynamics_2d",
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(ic_u, bc_no_flux),
        sample_defaults={"n_col": 80_000, "n_ic": 10_000, "n_bc": 8_000},
        scales=ScaleSpec(L=Lx, U=1.0, alpha=D),
        field_ranges={"u": (-1.0, 1.0)},
        references=(
            "Hegselmann, R., Krause, U. (2002). "
            "Opinion dynamics and bounded confidence: models, analysis and simulation. "
            "JASSS, 5(3).",
            "Allen, S.M., Cahn, J.W. (1979). "
            "A microscopic theory for antiphase boundary motion. "
            "Acta Metallurgica, 27(6), 1085-1095.",
        ),
        domain_bounds={"x": (0.0, Lx), "y": (0.0, Ly), "t": (0.0, t_end)},
        solver_spec={
            "name": "fdm",
            "method": "etd_rk4_2d",
            "params": {"nx": 128, "ny": 128, "nt": 500, "dt": t_end / 500.0},
        },
    )
