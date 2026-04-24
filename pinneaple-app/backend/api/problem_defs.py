"""
Problem library and physics utilities for the PINNeAPPle backend.
Self-contained — does not import from the Streamlit app.
"""
from __future__ import annotations
import os
import re
import json
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── Problem library ───────────────────────────────────────────────────────────

PROBLEMS: Dict[str, Dict] = {
    "burgers_1d": {
        "name":        "Burgers Equation (1D)",
        "category":    "Fluid Dynamics",
        "description": "1-D viscous Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²",
        "equations":   ["\\partial u/\\partial t + u \\partial u/\\partial x = \\nu \\partial^2 u/\\partial x^2"],
        "domain":      {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        "params":      {"nu": 0.01},
        "bcs":         ["u(-1,t)=0", "u(1,t)=0"],
        "ics":         ["u(x,0) = -sin(π x)"],
        "dim":         2,
        "tags":        ["pde", "nonlinear", "hyperbolic"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Raissi et al. (2019)",
    },
    "heat_2d": {
        "name":        "Heat Equation (2D steady)",
        "category":    "Heat Transfer",
        "description": "2-D steady-state heat conduction: ∇²T = -Q/k",
        "equations":   ["\\nabla^2 T = -Q/k"],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {"k": 1.0, "Q": 1.0},
        "bcs":         ["T=0 on all boundaries"],
        "ics":         [],
        "dim":         2,
        "tags":        ["pde", "elliptic", "linear"],
        "solvers":     ["fdm", "fem", "pinn"],
        "ref":         "Classic benchmark",
    },
    "poisson_2d": {
        "name":        "Poisson Equation (2D)",
        "category":    "Electromagnetics / Structural",
        "description": "∇²u = f(x,y) — prototypical elliptic PDE",
        "equations":   ["\\nabla^2 u = -2\\pi^2 \\sin(\\pi x) \\sin(\\pi y)"],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {},
        "bcs":         ["u=0 on ∂Ω"],
        "ics":         [],
        "dim":         2,
        "tags":        ["pde", "elliptic"],
        "solvers":     ["fdm", "fem", "pinn"],
        "ref":         "PINN benchmarks",
    },
    "navier_stokes_2d": {
        "name":        "Navier-Stokes (2D incompressible)",
        "category":    "Fluid Dynamics / CFD",
        "description": "Incompressible N-S: ∂u/∂t + (u·∇)u = -∇p + ν∇²u,  ∇·u=0",
        "equations":   [
            "\\partial u/\\partial t + u \\partial u/\\partial x + v \\partial u/\\partial y = -\\partial p/\\partial x + \\nu(\\partial^2 u/\\partial x^2 + \\partial^2 u/\\partial y^2)",
            "\\partial v/\\partial t + u \\partial v/\\partial x + v \\partial v/\\partial y = -\\partial p/\\partial y + \\nu(\\partial^2 v/\\partial x^2 + \\partial^2 v/\\partial y^2)",
            "\\partial u/\\partial x + \\partial v/\\partial y = 0",
        ],
        "domain":      {"x": [0.0, 2.0], "y": [0.0, 1.0]},
        "params":      {"nu": 0.01, "Re": 100},
        "bcs":         ["no-slip walls", "inlet: u=1", "outlet: p=0"],
        "ics":         ["u=v=0"],
        "dim":         3,
        "tags":        ["pde", "nonlinear", "parabolic", "cfd"],
        "solvers":     ["fvm", "lbm", "pinn"],
        "ref":         "Raissi et al. (2020)",
    },
    "wave_1d": {
        "name":        "Wave Equation (1D)",
        "category":    "Acoustics / Structural",
        "description": "∂²u/∂t² = c² ∂²u/∂x²",
        "equations":   ["\\partial^2 u/\\partial t^2 = c^2 \\partial^2 u/\\partial x^2"],
        "domain":      {"x": [0.0, 1.0], "t": [0.0, 1.0]},
        "params":      {"c": 1.0},
        "bcs":         ["u(0,t)=0", "u(1,t)=0"],
        "ics":         ["u(x,0)=sin(πx)", "∂u/∂t(x,0)=0"],
        "dim":         2,
        "tags":        ["pde", "hyperbolic"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Classic benchmark",
    },
    "lid_driven_cavity": {
        "name":        "Lid-Driven Cavity (LBM)",
        "category":    "CFD / LBM",
        "description": "Square cavity with top lid moving at u=u_in; Re=100–1000",
        "equations":   ["D2Q9 LBM — BGK collision with Zou-He BCs"],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {"Re": 100, "nx": 64, "ny": 64, "u_in": 0.05},
        "bcs":         ["top lid: u=u_in", "three walls: no-slip"],
        "ics":         [],
        "dim":         2,
        "tags":        ["cfd", "lbm", "benchmark"],
        "solvers":     ["lbm"],
        "ref":         "Ghia et al. (1982)",
    },
    "cylinder_flow": {
        "name":        "Flow Past Cylinder (LBM)",
        "category":    "CFD / LBM",
        "description": "Von Kármán vortex street — classic CFD benchmark",
        "equations":   ["D2Q9 LBM with cylinder bounce-back"],
        "domain":      {"x": [0.0, 4.0], "y": [0.0, 1.0]},
        "params":      {"Re": 200, "nx": 160, "ny": 64, "u_in": 0.05,
                        "obstacle": {"type": "cylinder", "cx": 40, "cy": 32, "r": 8}},
        "bcs":         ["velocity inlet", "pressure outlet", "no-slip cylinder"],
        "ics":         [],
        "dim":         2,
        "tags":        ["cfd", "lbm", "wake", "vortex"],
        "solvers":     ["lbm"],
        "ref":         "Schäfer & Turek (1996)",
    },
    "elastic_plate": {
        "name":        "Linear Elasticity (2D)",
        "category":    "Structural Mechanics",
        "description": "Plane-stress elasticity — σ_ij,j + b_i = 0",
        "equations":   [
            "\\partial \\sigma_{xx}/\\partial x + \\partial \\sigma_{xy}/\\partial y + b_x = 0",
            "\\partial \\sigma_{xy}/\\partial x + \\partial \\sigma_{yy}/\\partial y + b_y = 0",
        ],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {"E": 1e5, "nu": 0.3},
        "bcs":         ["fixed left edge", "traction on right edge"],
        "ics":         [],
        "dim":         2,
        "tags":        ["structural", "elasticity"],
        "solvers":     ["fem", "pinn"],
        "ref":         "Standard FEM benchmark",
    },
    "reaction_diffusion": {
        "name":        "Reaction-Diffusion (Gray-Scott)",
        "category":    "Chemical Engineering",
        "description": "Gray-Scott pattern formation: ∂u/∂t = D_u∇²u - uv² + F(1-u)",
        "equations":   [
            "\\partial u/\\partial t = D_u \\nabla^2 u - uv^2 + F(1-u)",
            "\\partial v/\\partial t = D_v \\nabla^2 v + uv^2 - (F+k)v",
        ],
        "domain":      {"x": [0.0, 2.5], "y": [0.0, 2.5], "t": [0.0, 10.0]},
        "params":      {"Du": 2e-5, "Dv": 1e-5, "F": 0.04, "k": 0.06},
        "bcs":         ["periodic"],
        "ics":         ["u≈1, v≈0 with small random perturbation"],
        "dim":         3,
        "tags":        ["parabolic", "nonlinear", "chemistry"],
        "solvers":     ["fdm", "pinn"],
        "ref":         "Gray & Scott (1984)",
    },
}

CATEGORIES: List[str] = sorted(set(p["category"] for p in PROBLEMS.values()))


def get_problem(key: str) -> Optional[Dict]:
    return PROBLEMS.get(key)


def list_problems(category: Optional[str] = None) -> List[Tuple[str, Dict]]:
    items = list(PROBLEMS.items())
    if category:
        items = [(k, v) for k, v in items if v["category"] == category]
    return items


# ── Collocation point generation ──────────────────────────────────────────────

def generate_collocation_points(problem: Dict, n_interior: int = 2000,
                                n_boundary: int = 400) -> Dict:
    domain = problem.get("domain", {"x": [0, 1], "y": [0, 1]})
    keys   = list(domain.keys())
    lo     = np.array([domain[k][0] for k in keys], dtype=np.float32)
    hi     = np.array([domain[k][1] for k in keys], dtype=np.float32)

    rng = np.random.default_rng(42)
    interior = rng.uniform(lo, hi, size=(n_interior, len(keys))).astype(np.float32)

    boundary_pts = []
    for dim_i in range(len(keys)):
        for val in [lo[dim_i], hi[dim_i]]:
            pts = rng.uniform(lo, hi,
                              size=(n_boundary // (2 * len(keys)), len(keys))).astype(np.float32)
            pts[:, dim_i] = val
            boundary_pts.append(pts)
    boundary = np.vstack(boundary_pts) if boundary_pts else interior[:n_boundary]

    return {
        "interior":    interior,
        "boundary":    boundary,
        "coord_names": keys,
        "domain":      domain,
    }


# ── PDE loss builders ─────────────────────────────────────────────────────────

def build_pinn_loss(problem: Dict):
    import torch
    import math as _math

    pde_kind = problem.get("_preset_key", "")
    params   = problem.get("params", {})
    nu       = float(params.get("nu",  0.01))
    c_wave   = float(params.get("c",   1.0))
    Q_src    = float(params.get("Q",   1.0))
    k_cond   = float(params.get("k",   1.0))

    def _grad(u, pts):
        return torch.autograd.grad(u.sum(), pts, create_graph=True)[0]

    def _laplacian_2d(u, pts):
        """∇²u = u_xx + u_yy for 2-D collocation pts (N, 2)."""
        g    = _grad(u, pts)
        u_xx = _grad(g[:, 0:1], pts)[:, 0:1]
        u_yy = _grad(g[:, 1:2], pts)[:, 1:2]
        return u_xx + u_yy

    # ── Burgers: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²  (coords: x=col0, t=col1)
    def burgers_residual(model, pts):
        pts  = pts.requires_grad_(True)
        u    = model(pts)
        g    = _grad(u, pts)
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = _grad(u_x, pts)[:, 0:1]
        return (u_t + u * u_x - nu * u_xx).pow(2).mean()

    # ── Poisson 2-D: ∇²u = -2π²·sin(πx)·sin(πy)  (exact solution: sin(πx)sin(πy))
    def poisson_residual(model, pts):
        pts  = pts.requires_grad_(True)
        u    = model(pts)
        lap  = _laplacian_2d(u, pts)
        f    = (-2.0 * _math.pi**2
                * torch.sin(_math.pi * pts[:, 0:1])
                * torch.sin(_math.pi * pts[:, 1:2]))
        return (lap - f).pow(2).mean()

    # ── Heat 2-D steady-state: ∇²T = -Q/k
    def heat_residual(model, pts):
        pts  = pts.requires_grad_(True)
        u    = model(pts)
        lap  = _laplacian_2d(u, pts)
        return (lap + Q_src / k_cond).pow(2).mean()

    # ── Wave 1-D: ∂²u/∂t² = c²·∂²u/∂x²  (coords: x=col0, t=col1)
    def wave_residual(model, pts):
        pts  = pts.requires_grad_(True)
        u    = model(pts)
        g    = _grad(u, pts)
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = _grad(u_x, pts)[:, 0:1]
        u_tt = _grad(u_t, pts)[:, 1:2]
        return (u_tt - c_wave**2 * u_xx).pow(2).mean()

    # ── Navier-Stokes 2-D (steady, incompressible, stream-function ψ formulation)
    # ∇⁴ψ = 0  for Stokes;  full NS: u·∂ω/∂x + v·∂ω/∂y = ν·∇²ω  where ω=-∇²ψ
    def ns_residual(model, pts):
        pts  = pts.requires_grad_(True)
        psi  = model(pts)
        g    = _grad(psi, pts)
        psi_x, psi_y = g[:, 0:1], g[:, 1:2]
        # Vorticity: ω = -∇²ψ
        psi_xx = _grad(psi_x, pts)[:, 0:1]
        psi_yy = _grad(psi_y, pts)[:, 1:2]
        omega  = -(psi_xx + psi_yy)
        # Convective vorticity transport residual: u·∂ω/∂x + v·∂ω/∂y - ν·∇²ω = 0
        # u = ∂ψ/∂y, v = -∂ψ/∂x
        u_vel  = psi_y    # (N,1)
        v_vel  = -psi_x   # (N,1)
        g_om   = _grad(omega, pts)
        omega_x, omega_y = g_om[:, 0:1], g_om[:, 1:2]
        omega_xx = _grad(omega_x, pts)[:, 0:1]
        omega_yy = _grad(omega_y, pts)[:, 1:2]
        res = u_vel * omega_x + v_vel * omega_y - nu * (omega_xx + omega_yy)
        return res.pow(2).mean()

    # ── Generic Laplace fallback: ∇²u = 0
    def laplace_residual(model, pts):
        pts  = pts.requires_grad_(True)
        u    = model(pts)
        lap  = _laplacian_2d(u, pts)
        return lap.pow(2).mean()

    dispatch = {
        "burgers_1d":        burgers_residual,
        "poisson_2d":        poisson_residual,
        "heat_2d":           heat_residual,
        "wave_1d":           wave_residual,
        "navier_stokes_2d":  ns_residual,
    }
    if pde_kind in dispatch:
        return dispatch[pde_kind]
    # substring fallback so keyword-matched variants also hit the right residual
    if "burgers" in pde_kind:
        return burgers_residual
    if "poisson" in pde_kind:
        return poisson_residual
    if "heat" in pde_kind:
        return heat_residual
    if "wave" in pde_kind:
        return wave_residual
    return laplace_residual


# ── AI formulation ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a physics expert integrated into the PINNeAPPle Physics-AI platform.

Given a natural-language description of a physical problem, return a JSON object with:
  - name            (string)
  - category        (string — one of: Fluid Dynamics, Heat Transfer, Structural Mechanics,
                     Acoustics, Electromagnetics, Chemical Engineering, Other)
  - description     (string — concise)
  - equations       (list of strings — governing equations in LaTeX notation)
  - domain          (dict — coordinate → [min, max])
  - params          (dict — key physical parameters with typical values)
  - bcs             (list of strings — boundary conditions)
  - ics             (list of strings — initial conditions, empty if steady)
  - dim             (int — number of independent variables)
  - tags            (list of strings)
  - solvers         (list — suggested: fdm, fem, fvm, lbm, pinn, fno, deeponet)
  - ref             (string — relevant reference if well-known)

Return ONLY valid JSON, no markdown fences, no explanations.
"""

_KEYWORD_MAP = {
    ("navier", "stokes"):          "navier_stokes_2d",
    ("flow", "cylinder"):          "cylinder_flow",
    ("cavity", "lid"):             "lid_driven_cavity",
    ("burgers",):                  "burgers_1d",
    ("heat", "conduc"):            "heat_2d",
    ("wave",):                     "wave_1d",
    ("acoustic",):                 "wave_1d",
    ("elastic", "plate"):          "elastic_plate",
    ("reaction", "diffusion"):     "reaction_diffusion",
    ("gray", "scott"):             "reaction_diffusion",
    ("poisson",):                  "poisson_2d",
}


def _keyword_match(description: str) -> Optional[str]:
    low = description.lower()
    for kws, key in _KEYWORD_MAP.items():
        if all(kw in low for kw in kws):
            return key
    return None


def formulate_with_ai(description: str) -> Dict:
    preset_key = _keyword_match(description)
    if preset_key:
        prob = get_problem(preset_key)
        if prob:
            return {**prob, "_source": "keyword_match", "_preset_key": preset_key}

    from django.conf import settings
    api_key = settings.ANTHROPIC_API_KEY
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": description}],
            )
            raw  = msg.content[0].text.strip()
            spec = json.loads(raw)
            spec["_source"] = "claude_api"
            return spec
        except Exception:
            pass

    return {
        "name":        "Custom Problem",
        "category":    "Other",
        "description": description,
        "equations":   ["[Equations not identified — describe more precisely]"],
        "domain":      {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "params":      {},
        "bcs":         ["[Define boundary conditions]"],
        "ics":         [],
        "dim":         2,
        "tags":        ["custom"],
        "solvers":     ["pinn"],
        "ref":         "",
        "_source":     "fallback",
    }


def suggest_models(problem: Dict) -> List[Dict]:
    tags    = problem.get("tags", [])
    solvers = problem.get("solvers", [])
    dim     = problem.get("dim", 2)
    out     = []

    if "lbm" in solvers:
        out.append({"model": "LBM Solver", "type": "lbm",
                    "reason": "Native D2Q9 LBM for this CFD problem", "score": 95})
    if "pinn" in solvers or True:
        out.append({"model": "PINN (MLP)", "type": "pinn_mlp",
                    "reason": "Physics-Informed Neural Network — data-efficient", "score": 85})
    if dim >= 2 and "fdm" in solvers:
        out.append({"model": "FDM Solver", "type": "fdm",
                    "reason": "Finite Difference — fast for structured grids", "score": 80})
    if "fem" in solvers:
        out.append({"model": "FEM Solver", "type": "fem",
                    "reason": "Finite Element — handles complex geometry", "score": 78})
    if "nonlinear" in tags:
        out.append({"model": "DeepONet", "type": "deeponet",
                    "reason": "Operator learning for nonlinear dynamics", "score": 75})

    return sorted(out, key=lambda x: -x["score"])
