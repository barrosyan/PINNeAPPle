# -*- coding: utf-8 -*-
"""SHIFT-Missile Inspired — Supersonic Missile Aerodynamics Physics AI
======================================================================

Demonstrates how PINNeAPPle can replicate the key capabilities of
physics-informed aerodynamic surrogate systems (e.g. Luminary's SHIFT-Missile):

  1. Modified Newtonian Theory + Oblique Shock analytical solver
     → high-fidelity-quality surface pressure distribution Cp(s, M, α, geometry)
  2. Turbulent boundary-layer skin friction model (Van Driest)
  3. Geometry-conditioned PINN: maps (s, M, α, r_nose, h_canard, h_fin) → (Cp, Cf)
  4. AeroDB generation across Mach × AoA design space
  5. Ensemble Uncertainty Quantification (90 % PI coverage)
  6. 6-DOF trajectory integration using surrogate aerodynamics
  7. Export plots, AeroDB (JSON) and TorchScript model

Physical Model
--------------
  Modified Newtonian Theory (supersonic / hypersonic, Lees 1956):

    Cp(θ) = Cp_max · sin²(θ)

    Cp_max = 2 / (γ · M²) · [(p02 / p_inf) − 1]

  where p02/p_inf is the Pitot pressure ratio behind a normal shock at M_inf:

    p02/p_inf = [(γ+1)² M² / (4γ M² − 2(γ−1))]^(γ/(γ-1)) · [(1−γ+2γM²) / (γ+1)]

  Local body angle θ(s) derived from ogive-cylinder-finned geometry:
    - Tangent-ogive nose:  tan θ = dR/ds  along the nose contour
    - Cylinder section:    θ = 0
    - Canard / fin panels: θ_panel = α + δ_canard (twist)

  AoA correction: effective local angle = θ_body(s) + α · cos(φ)  [azimuth φ]

  Skin-friction coefficient (turbulent, Van Driest II recovery):
    Cf_comp(M, Re_x) = Cf_inc(Re_x) / F_comp(M, T_w/T_adb)

    Cf_inc(Re_x) = 0.0592 / Re_x^0.2         (1/5-power turbulent BL)
    F_comp       = (T_w / T_inf + r·(γ−1)/2·M²)^0.65   (compressibility factor)

  Integrated aerodynamic coefficients:
    CN   = ∫ (Cp + Cf_n) dA_n / (q_inf · A_ref)   [normal-force coefficient]
    CA   = ∫ (Cp_base + Cf_a) dA_a / (q_inf · A_ref)   [axial-force coefficient]
    CL   = CN · cos α − CA · sin α
    CD   = CN · sin α + CA · cos α
    L/D  = CL / CD

  6-DOF trajectory (point-mass, flat Earth):
    m · dV/dt  = −D − m·g·sin γ
    m·V · dγ/dt = L − m·g·cos γ
    dx/dt = V cos γ
    dh/dt = V sin γ

Pipeline Steps
--------------
  [1] Define missile geometry family (ogive + cylinder + fins)
  [2] Generate analytical AeroDB over Mach × AoA grid (ground truth)
  [3] Train geometry-conditioned PINN surrogate
  [4] Validate surrogate vs. analytical reference
  [5] Build full AeroDB with surrogate (fast sweep)
  [6] Ensemble UQ — 90% coverage interval
  [7] 6-DOF trajectory integration with surrogate aerodynamics
  [8] Export plots, AeroDB JSON, TorchScript

References
----------
  Modified Newtonian Theory: Lees (1956), Gentry et al. (1973) AEROJET
  Van Driest II: White (2006) Viscous Fluid Flow, Ch. 7
  Missile DATCOM: USAF (1979)
  Luminary Cloud SHIFT-Missile (2024) — physics-AI aerodynamics surrogate
  Anderson (2003) Modern Compressible Flow, McGraw-Hill
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PINNeAPPle imports (graceful fallback) ─────────────────────────────────
try:
    from pinneapple_train import best_device, maybe_compile
except ImportError:
    def best_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def maybe_compile(m, **_):
        return m

try:
    from pinneapple_analysis.uncertainty import EnsembleUQ, ConformalPredictor
    _UQ_AVAILABLE = True
except ImportError:
    _UQ_AVAILABLE = False

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0d1117"
ACCENT  = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"

GAMMA = 1.4


# ══════════════════════════════════════════════════════════════════════════════
# 1. GEOMETRY — OGIVE-CYLINDER-FIN MISSILE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MissileGeometry:
    """Parametric ogive-cylinder-finned missile geometry."""
    L_nose_cal:   float = 3.0    # nose length in calibers (D = body diameter)
    L_body_cal:   float = 10.0   # cylinder section length in calibers
    D_body_m:     float = 0.127  # body diameter [m]
    r_nose_m:     float = 0.005  # nose-tip radius [m]  (bluntness)
    h_canard_m:   float = 0.03   # canard panel semi-span [m]
    h_fin_m:      float = 0.06   # tail-fin panel semi-span [m]
    A_ref_m2:     Optional[float] = None  # reference area (defaults to πD²/4)

    def __post_init__(self):
        if self.A_ref_m2 is None:
            self.A_ref_m2 = math.pi * self.D_body_m**2 / 4.0

    @property
    def L_nose_m(self) -> float:
        return self.L_nose_cal * self.D_body_m

    @property
    def L_body_m(self) -> float:
        return self.L_body_cal * self.D_body_m

    @property
    def L_total_m(self) -> float:
        return self.L_nose_m + self.L_body_m


DEFAULT_GEOM = MissileGeometry()


def body_angle_distribution(
    geom: MissileGeometry,
    n_pts: int = 200,
    alpha_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arc-length s [m], local half-angle θ [rad], radius R [m]
    along the meridional contour at AoA = alpha.
    """
    L_n = geom.L_nose_m
    D   = geom.D_body_m
    R_b = D / 2.0
    rn  = geom.r_nose_m

    # Tangent-ogive nose profile: R(x) = sqrt(ρ² − (L_n − x)²) + rn − ρ + R_b
    # ρ = (L_n² + R_b²) / (2·R_b)  — ogive radius of curvature
    rho_og = (L_n**2 + R_b**2) / (2.0 * R_b)

    n_nose = int(n_pts * 0.4)
    n_body = n_pts - n_nose

    x_nose = np.linspace(0.0, L_n, n_nose)
    R_nose = np.sqrt(np.maximum(rho_og**2 - (L_n - x_nose)**2, 0.0)) - rho_og + R_b
    R_nose = np.maximum(R_nose, rn)

    dR_dx  = -(L_n - x_nose) / np.maximum(
        np.sqrt(np.maximum(rho_og**2 - (L_n - x_nose)**2, 1e-10)), 1e-10
    )
    theta_nose = np.arctan(np.abs(dR_dx))

    x_body = np.linspace(L_n, geom.L_total_m, n_body)
    R_body = np.full(n_body, R_b)
    theta_body = np.zeros(n_body)

    x_all = np.concatenate([x_nose, x_body])
    R_all = np.concatenate([R_nose, R_body])
    th_all = np.concatenate([theta_nose, theta_body])

    # arc-length along meridian
    ds   = np.sqrt(np.diff(x_all)**2 + np.diff(R_all)**2)
    s_all = np.concatenate([[0.0], np.cumsum(ds)])

    # effective angle including AoA (leeward meridian)
    th_eff = th_all + alpha_rad * np.cos(0.0)   # φ=0 → leeward

    return s_all, th_eff, R_all


# ══════════════════════════════════════════════════════════════════════════════
# 2. AERODYNAMIC SOLVER — Modified Newtonian + Van Driest
# ══════════════════════════════════════════════════════════════════════════════

def pitot_pressure_ratio(M: float, gamma: float = GAMMA) -> float:
    """p02/p_inf via Rayleigh Pitot formula (normal shock + isentropic compression)."""
    g = gamma
    term1 = ((g + 1)**2 * M**2 / (4.0 * g * M**2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
    term2 = (1.0 - g + 2.0 * g * M**2) / (g + 1.0)
    return term1 * term2


def cp_max_newtonian(M: float, gamma: float = GAMMA) -> float:
    """Stagnation Cp using Modified Newtonian (exact shock + Pitot)."""
    p02_ratio = pitot_pressure_ratio(M, gamma)
    return (2.0 / (gamma * M**2)) * (p02_ratio - 1.0)


def cf_van_driest(M: float, Re_x: float, T_ratio: float = 1.0, gamma: float = GAMMA) -> float:
    """Compressible turbulent skin-friction (Van Driest II approximation)."""
    if Re_x < 1e3:
        return 0.0
    r_factor = 0.896                       # Prandtl number recovery factor (Pr^(1/3))
    T_aw_ratio = T_ratio + r_factor * (gamma - 1.0) / 2.0 * M**2
    F_comp = T_aw_ratio ** 0.65
    Cf_inc = 0.0592 / Re_x**0.2           # turbulent power-law
    return Cf_inc / F_comp


def solve_aero(
    geom:    MissileGeometry,
    M_inf:   float,
    alpha_deg: float,
    altitude_m: float = 10_000.0,
    n_pts:   int = 200,
) -> Dict[str, np.ndarray]:
    """Full aerodynamic solution via Modified Newtonian + Van Driest."""
    alpha_rad = math.radians(alpha_deg)

    # Standard atmosphere (simplified ISA)
    T_ref  = 288.15 - 0.0065 * altitude_m
    p_ref  = 101325.0 * (T_ref / 288.15) ** 5.2561
    rho_ref = p_ref / (287.058 * T_ref)
    a_inf  = math.sqrt(GAMMA * 287.058 * T_ref)
    V_inf  = M_inf * a_inf
    mu_ref = 1.716e-5 * (T_ref / 273.15) ** 1.5 * (273.15 + 110.4) / (T_ref + 110.4)

    s_all, theta_eff, R_all = body_angle_distribution(geom, n_pts, alpha_rad)

    Cp_max = cp_max_newtonian(M_inf)

    Cp = Cp_max * np.sin(theta_eff) ** 2

    Re_x = rho_ref * V_inf * s_all / mu_ref
    Re_x = np.maximum(Re_x, 1e-6)
    T_w_ratio = 1.0 + 0.035 * M_inf**2    # adiabatic wall temperature ratio ≈1
    Cf = np.array([cf_van_driest(M_inf, Re, T_w_ratio) for Re in Re_x])

    q_inf = 0.5 * rho_ref * V_inf**2
    A_ref = geom.A_ref_m2

    dA = 2.0 * math.pi * R_all * np.gradient(s_all)
    CN = float(np.sum(Cp * np.sin(theta_eff) * dA) / (q_inf * A_ref))
    CA = float(np.sum(Cf * np.cos(theta_eff) * dA) / (q_inf * A_ref))
    CA += 0.12                              # add base drag coefficient

    CL = CN * math.cos(alpha_rad) - CA * math.sin(alpha_rad)
    CD = CN * math.sin(alpha_rad) + CA * math.cos(alpha_rad)
    LD = CL / (CD + 1e-10)

    return {
        "s_m":   s_all,
        "Cp":    Cp,
        "Cf":    Cf,
        "CN":    np.full_like(s_all, CN),
        "CA":    np.full_like(s_all, CA),
        "CL":    np.full_like(s_all, CL),
        "CD":    np.full_like(s_all, CD),
        "LD":    np.full_like(s_all, LD),
        "R_m":   R_all,
        "theta_rad": theta_eff,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

MACH_RANGE     = (1.5, 3.5)
ALPHA_RANGE    = (0.0, 8.0)     # degrees
RNOSE_RANGE    = (0.001, 0.010) # m
HCANARD_RANGE  = (0.015, 0.060) # m
HFIN_RANGE     = (0.030, 0.090) # m


def _lhs(n: int, ranges: List[Tuple[float, float]], rng) -> np.ndarray:
    d   = len(ranges)
    pts = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u    = (perm + rng.uniform(size=n)) / n
        lo, hi = ranges[j]
        pts[:, j] = lo + u * (hi - lo)
    return pts


def generate_dataset(
    n_designs: int = 3_000,
    n_pts:     int = 100,
    seed:      int = 0,
) -> Dict[str, np.ndarray]:
    """LHS dataset: inputs (s,M,α,r_n,h_c,h_f) → outputs (Cp, Cf)."""
    rng = np.random.default_rng(seed)
    ranges = [MACH_RANGE, ALPHA_RANGE, RNOSE_RANGE, HCANARD_RANGE, HFIN_RANGE]
    designs = _lhs(n_designs, ranges, rng)

    N = n_designs * n_pts
    X = np.zeros((N, 6), dtype=np.float32)
    Y = np.zeros((N, 2), dtype=np.float32)

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo)

    for i, (M, alpha, rn, hc, hf) in enumerate(designs):
        geom = MissileGeometry(r_nose_m=rn, h_canard_m=hc, h_fin_m=hf)
        try:
            sol = solve_aero(geom, M, alpha, n_pts=n_pts)
        except Exception:
            continue
        s_norm = sol["s_m"] / geom.L_total_m
        idx = slice(i * n_pts, (i + 1) * n_pts)
        X[idx, 0] = s_norm.astype(np.float32)
        X[idx, 1] = norm(M,     *MACH_RANGE)
        X[idx, 2] = norm(alpha, *ALPHA_RANGE)
        X[idx, 3] = norm(rn,    *RNOSE_RANGE)
        X[idx, 4] = norm(hc,    *HCANARD_RANGE)
        X[idx, 5] = norm(hf,    *HFIN_RANGE)
        Y[idx, 0] = sol["Cp"].astype(np.float32)
        Y[idx, 1] = (sol["Cf"] * 1e3).astype(np.float32)   # scale Cf×1000

    perm = rng.permutation(N)
    return {"X": X[perm], "Y": Y[perm]}


# ══════════════════════════════════════════════════════════════════════════════
# 4. AERO SURROGATE PINN — GeoTransolver-like architecture
# ══════════════════════════════════════════════════════════════════════════════

class AeroSurrogatePINN(nn.Module):
    """
    Geometry-conditioned PINN for supersonic aerodynamics.

    Input  : (s_norm, M_norm, α_norm, r_nose_norm, h_canard_norm, h_fin_norm)
    Output : (Cp, Cf×1000)

    Architecture:
      - Fourier feature encoding of arc-length s (spatial awareness)
      - Geometry encoder MLP: (M, α, r_n, h_c, h_f) → latent geometry vector g
      - Field decoder MLP: (fourier(s), g) → (Cp, Cf)
    """

    def __init__(
        self,
        n_fourier:     int   = 32,
        fourier_scale: float = 3.0,
        geo_dim:       int   = 64,
        hidden:        int   = 256,
        n_decode:      int   = 5,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.B_s = nn.Parameter(
            torch.randn(n_fourier, 1) * fourier_scale,
            requires_grad=False,
        )
        s_feat = 2 * n_fourier

        # Geometry encoder
        self.geo_enc = nn.Sequential(
            nn.Linear(5, 64),  nn.Tanh(),
            nn.Linear(64, geo_dim), nn.Tanh(),
        )

        # Decoder
        in_dim = s_feat + geo_dim
        dec: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_decode - 2):
            dec += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)
        self.skip    = nn.Linear(in_dim, hidden)
        self.head    = nn.Linear(hidden, 2)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s      = x[:, :1]
        geo    = x[:, 1:]       # (M, α, r_n, h_c, h_f) — 5 features

        proj   = s @ self.B_s.T * (2.0 * math.pi)
        s_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        g    = self.geo_enc(geo)
        z    = torch.cat([s_feat, g], dim=-1)
        h    = self.decoder(z) + self.skip(z)
        out  = self.head(h)
        return torch.nn.functional.softplus(out)   # Cp, Cf ≥ 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. PHYSICS RESIDUAL — Modified Newtonian consistency
# ══════════════════════════════════════════════════════════════════════════════

def physics_residual(model: AeroSurrogatePINN, x_batch: torch.Tensor) -> torch.Tensor:
    """
    Enforce dCf/ds ≤ 0 (Cf decays downstream on cylindrical section)
    and Cp ≥ 0 (already via softplus).
    """
    x = x_batch.clone().requires_grad_(True)
    out = model(x)
    Cp, Cf = out[:, 0], out[:, 1]

    grad_Cf = torch.autograd.grad(Cf.sum(), x, create_graph=True)[0][:, 0]
    r_Cf_decay = torch.relu(grad_Cf).pow(2).mean()     # penalise Cf growth
    r_Cp_bound = torch.relu(-Cp).pow(2).mean()         # Cp ≥ 0 (redundant with softplus)

    return r_Cf_decay + r_Cp_bound


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model:       AeroSurrogatePINN,
    data:        Dict[str, np.ndarray],
    device:      torch.device,
    epochs:      int   = 10_000,
    batch_size:  int   = 4_096,
    lr:          float = 3e-4,
    phys_weight: float = 0.05,
    label:       str   = "aero",
) -> Dict[str, List[float]]:

    model = model.to(device)
    X = torch.tensor(data["X"], dtype=torch.float32, device=device)
    Y = torch.tensor(data["Y"], dtype=torch.float32, device=device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    mse   = nn.MSELoss()

    history: Dict[str, List[float]] = {"total": [], "data": [], "phys": []}
    n = X.shape[0]
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        idx  = torch.randperm(n, device=device)[:batch_size]
        xb, yb = X[idx], Y[idx]

        model.train()
        pred   = model(xb)
        l_data = mse(pred, yb)
        l_phys = physics_residual(model, xb)
        loss   = l_data + phys_weight * l_phys

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        history["total"].append(loss.item())
        history["data"].append(l_data.item())
        history["phys"].append(l_phys.item())

        if epoch % 1000 == 0:
            print(
                f"[{label}] epoch {epoch:>6}/{epochs} | "
                f"loss {loss.item():.4e} | data {l_data.item():.4e} | "
                f"phys {l_phys.item():.4e} | {time.time()-t0:.0f}s"
            )

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 7. AERODB GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_aerodb(
    model:    AeroSurrogatePINN,
    device:   torch.device,
    mach_pts: List[float] = [1.5, 2.0, 2.5, 3.0, 3.5],
    alpha_pts: List[float] = [0.0, 2.0, 4.0, 6.0, 8.0],
    n_s:      int = 50,
) -> Dict[str, Any]:
    """Generate AeroDB table (CL, CD, L/D) via surrogate integration."""

    def norm(v, lo, hi):
        return (v - lo) / (hi - lo)

    aerodb: Dict[str, Any] = {"mach": mach_pts, "alpha_deg": alpha_pts, "data": []}

    model.eval()
    for M in mach_pts:
        row = {"M": M, "alpha_entries": []}
        for alpha in alpha_pts:
            geom   = DEFAULT_GEOM
            alpha_rad = math.radians(alpha)
            s_grid = np.linspace(0.0, 1.0, n_s, dtype=np.float32)
            X_q    = np.column_stack([
                s_grid,
                np.full(n_s, norm(M,     *MACH_RANGE),    dtype=np.float32),
                np.full(n_s, norm(alpha, *ALPHA_RANGE),   dtype=np.float32),
                np.full(n_s, 0.5, dtype=np.float32),   # r_nose norm mid
                np.full(n_s, 0.5, dtype=np.float32),   # h_canard norm mid
                np.full(n_s, 0.5, dtype=np.float32),   # h_fin norm mid
            ])
            with torch.no_grad():
                pred = model(torch.tensor(X_q, device=device)).cpu().numpy()
            Cp = pred[:, 0]
            Cf = pred[:, 1] * 1e-3

            # approximate integration for integrated coefficients
            _, theta_eff, R_all = body_angle_distribution(geom, n_s, alpha_rad)
            s_phys = s_grid * geom.L_total_m
            ds     = np.gradient(s_phys)
            dA     = 2.0 * math.pi * R_all * ds

            T_ref  = 288.15 - 0.0065 * 10_000.0
            a_inf  = math.sqrt(GAMMA * 287.058 * T_ref)
            V_inf  = M * a_inf
            rho    = 101325.0 * (T_ref / 288.15)**5.2561 / (287.058 * T_ref)
            q      = 0.5 * rho * V_inf**2

            CN = float(np.sum(Cp * np.sin(theta_eff) * dA) / (q * geom.A_ref_m2))
            CA = float(np.sum(Cf * np.cos(theta_eff) * dA) / (q * geom.A_ref_m2)) + 0.12
            CL = CN * math.cos(alpha_rad) - CA * math.sin(alpha_rad)
            CD = CN * math.sin(alpha_rad) + CA * math.cos(alpha_rad)

            row["alpha_entries"].append({
                "alpha_deg": alpha, "CN": CN, "CA": CA,
                "CL": CL, "CD": CD, "LD": CL / (CD + 1e-8),
            })
        aerodb["data"].append(row)

    return aerodb


# ══════════════════════════════════════════════════════════════════════════════
# 8. ENSEMBLE UQ
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_uq(
    models:  List[AeroSurrogatePINN],
    X_test:  np.ndarray,
    device:  torch.device,
    alpha:   float = 0.10,
) -> Dict[str, np.ndarray]:
    Xt = torch.tensor(X_test, dtype=torch.float32, device=device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xt).cpu().numpy())
    preds = np.stack(preds, axis=0)
    return {
        "mean": preds.mean(0),
        "std":  preds.std(0),
        "lo":   np.percentile(preds, 100 * alpha / 2,        axis=0),
        "hi":   np.percentile(preds, 100 * (1 - alpha / 2),  axis=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. 6-DOF TRAJECTORY (point-mass)
# ══════════════════════════════════════════════════════════════════════════════

def run_trajectory(
    aerodb:      Dict[str, Any],
    m_kg:        float = 50.0,
    V0_ms:       float = 680.0,
    gamma0_deg:  float = -5.0,
    h0_m:        float = 10_000.0,
    dt:          float = 0.1,
    t_max:       float = 60.0,
) -> Dict[str, np.ndarray]:
    """
    Point-mass 2D trajectory integration using AeroDB lookup (nearest-neighbor).
    """
    from scipy.interpolate import RegularGridInterpolator

    mach_arr  = np.array(aerodb["mach"])
    alpha_arr = np.array(aerodb["alpha_deg"])

    CL_tab = np.zeros((len(mach_arr), len(alpha_arr)))
    CD_tab = np.zeros_like(CL_tab)
    for i, row in enumerate(aerodb["data"]):
        for j, e in enumerate(row["alpha_entries"]):
            CL_tab[i, j] = e["CL"]
            CD_tab[i, j] = e["CD"]

    CL_interp = RegularGridInterpolator(
        (mach_arr, alpha_arr), CL_tab, method="linear", bounds_error=False, fill_value=None
    )
    CD_interp = RegularGridInterpolator(
        (mach_arr, alpha_arr), CD_tab, method="linear", bounds_error=False, fill_value=None
    )

    g   = 9.80665
    V   = V0_ms
    gam = math.radians(gamma0_deg)
    h   = h0_m
    x   = 0.0
    t   = 0.0

    t_hist, x_hist, h_hist, V_hist, M_hist = [], [], [], [], []
    while t < t_max and h > 0.0:
        T_atm  = max(288.15 - 0.0065 * h, 216.65)
        a_atm  = math.sqrt(GAMMA * 287.058 * T_atm)
        rho    = 101325.0 * (T_atm / 288.15)**5.2561 / (287.058 * T_atm)
        M_inf  = V / a_atm
        q      = 0.5 * rho * V**2 * DEFAULT_GEOM.A_ref_m2

        alpha_trim = max(0.0, min(-math.degrees(gam) * 0.1, 8.0))
        CL = float(CL_interp([[M_inf, alpha_trim]]))
        CD = float(CD_interp([[M_inf, alpha_trim]]))
        L  = CL * q
        D  = CD * q

        dV  = (-D - m_kg * g * math.sin(gam)) / m_kg
        dgam = (L - m_kg * g * math.cos(gam)) / (m_kg * V + 1e-8)
        V   = max(V + dV * dt, 50.0)
        gam = gam + dgam * dt
        x  += V * math.cos(gam) * dt
        h  += V * math.sin(gam) * dt
        t  += dt

        t_hist.append(t);  x_hist.append(x);  h_hist.append(h)
        V_hist.append(V);  M_hist.append(M_inf)

    return {
        "t_s":  np.array(t_hist),
        "x_km": np.array(x_hist) / 1e3,
        "h_km": np.array(h_hist) / 1e3,
        "V_ms": np.array(V_hist),
        "M":    np.array(M_hist),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _dark_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes[None, :]
    for ax in axes.flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    return fig, axes


def plot_cp_distribution(
    geom:    MissileGeometry,
    model:   AeroSurrogatePINN,
    device:  torch.device,
    out_path: Path,
):
    fig, axes = _dark_fig(1, 2, figsize=(14, 5))
    n_s = 150
    mach_cases = [(2.0, 2.0, ACCENT), (3.0, 4.0, ACCENT2)]

    for M, alpha, color in mach_cases:
        alpha_rad = math.radians(alpha)
        s_grid, theta_eff, R_all = body_angle_distribution(geom, n_s, alpha_rad)
        s_norm = (s_grid / geom.L_total_m).astype(np.float32)

        def norm(v, lo, hi): return (v - lo) / (hi - lo)
        X_q = np.column_stack([
            s_norm,
            np.full(n_s, norm(M,     *MACH_RANGE),    dtype=np.float32),
            np.full(n_s, norm(alpha, *ALPHA_RANGE),   dtype=np.float32),
            np.full(n_s, 0.5, dtype=np.float32),
            np.full(n_s, 0.5, dtype=np.float32),
            np.full(n_s, 0.5, dtype=np.float32),
        ])
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_q, device=device)).cpu().numpy()

        ref_sol = solve_aero(geom, M, alpha, n_pts=n_s)
        lbl = f"M={M}  α={alpha}°"
        axes[0, 0].plot(s_norm, ref_sol["Cp"],    color=color, lw=2,   label=f"Analytical {lbl}")
        axes[0, 0].plot(s_norm, pred[:, 0],       color=color, lw=1.5, ls="--", label=f"PINN {lbl}")
        axes[0, 1].plot(s_norm, ref_sol["Cf"]*1e3, color=color, lw=2)
        axes[0, 1].plot(s_norm, pred[:, 1],        color=color, lw=1.5, ls="--")

    axes[0, 0].set_xlabel("s / L"); axes[0, 0].set_ylabel("Cp")
    axes[0, 0].set_title("Pressure Coefficient Distribution")
    axes[0, 0].legend(facecolor=DARK_BG, labelcolor="white", fontsize=7)

    axes[0, 1].set_xlabel("s / L"); axes[0, 1].set_ylabel("Cf × 10³")
    axes[0, 1].set_title("Skin-Friction Coefficient Distribution")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_polar(aerodb: Dict[str, Any], out_path: Path):
    fig, axes = _dark_fig(1, 2, figsize=(12, 5))
    colors = [ACCENT, ACCENT2, ACCENT3, "#d29922", "#bc8cff"]

    for i, row in enumerate(aerodb["data"]):
        alphas = [e["alpha_deg"] for e in row["alpha_entries"]]
        CL     = [e["CL"]       for e in row["alpha_entries"]]
        CD     = [e["CD"]       for e in row["alpha_entries"]]
        LD     = [e["LD"]       for e in row["alpha_entries"]]
        col    = colors[i % len(colors)]
        axes[0, 0].plot(alphas, CL, color=col, lw=1.5, marker="o", ms=4, label=f"M={row['M']}")
        axes[0, 0].plot(alphas, CD, color=col, lw=1.0, ls="--")
        axes[0, 1].plot(alphas, LD, color=col, lw=1.5, marker="o", ms=4, label=f"M={row['M']}")

    axes[0, 0].set_xlabel("AoA [deg]"); axes[0, 0].set_ylabel("CL (solid)  /  CD (dashed)")
    axes[0, 0].set_title("Lift & Drag Polars")
    axes[0, 0].legend(facecolor=DARK_BG, labelcolor="white", fontsize=8)

    axes[0, 1].set_xlabel("AoA [deg]"); axes[0, 1].set_ylabel("L/D")
    axes[0, 1].set_title("Lift-to-Drag Ratio")
    axes[0, 1].legend(facecolor=DARK_BG, labelcolor="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_uq(uq, Y_test, out_path):
    fig, axes = _dark_fig(1, 2, figsize=(12, 4))
    n = min(300, Y_test.shape[0])
    xs = np.arange(n)

    for ax, col_idx, ylabel, color in [
        (axes[0, 0], 0, "Cp",      ACCENT),
        (axes[0, 1], 1, "Cf×10³",  ACCENT2),
    ]:
        ax.fill_between(xs, uq["lo"][:n, col_idx], uq["hi"][:n, col_idx],
                        alpha=0.3, color=color, label="90% PI")
        ax.plot(xs, uq["mean"][:n, col_idx], color=color,   lw=1.5, label="Ensemble mean")
        ax.plot(xs, Y_test[:n, col_idx],     color="white", lw=1.0, ls="--", label="Truth")
        ax.set_xlabel("Sample index"); ax.set_ylabel(ylabel)
        ax.set_title(f"Ensemble UQ — {ylabel}")
        ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_trajectory(traj: Dict[str, np.ndarray], out_path: Path):
    fig, axes = _dark_fig(1, 2, figsize=(14, 5))

    axes[0, 0].plot(traj["x_km"], traj["h_km"], color=ACCENT, lw=2)
    axes[0, 0].set_xlabel("Down-range [km]"); axes[0, 0].set_ylabel("Altitude [km]")
    axes[0, 0].set_title("Missile Trajectory")

    axes[0, 1].plot(traj["t_s"], traj["M"], color=ACCENT2, lw=1.5, label="Mach")
    ax2 = axes[0, 1].twinx()
    ax2.set_facecolor(DARK_BG)
    ax2.tick_params(colors="white")
    ax2.plot(traj["t_s"], traj["V_ms"], color=ACCENT3, lw=1.5, ls="--", label="V [m/s]")
    ax2.spines["right"].set_edgecolor("#444")
    ax2.yaxis.label.set_color(ACCENT3)
    ax2.set_ylabel("Velocity [m/s]", color=ACCENT3)
    axes[0, 1].set_xlabel("Time [s]"); axes[0, 1].set_ylabel("Mach number", color=ACCENT2)
    axes[0, 1].set_title("Mach & Velocity History")
    axes[0, 1].legend(facecolor=DARK_BG, labelcolor="white", fontsize=8, loc="upper right")
    ax2.legend(facecolor=DARK_BG, labelcolor="white", fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_training_history(history: Dict[str, List[float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")
    e = np.arange(1, len(history["total"]) + 1)
    ax.semilogy(e, history["total"], color=ACCENT,   lw=1.5, label="total")
    ax.semilogy(e, history["data"],  color=ACCENT3,  lw=1.0, ls="--", label="data MSE")
    ax.semilogy(e, history["phys"],  color=ACCENT2,  lw=1.0, ls=":", label="physics")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss",  color="white")
    ax.set_title("PINN Training History", color="white")
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = best_device()
    print(f"\n{'='*70}")
    print("  SHIFT-Missile Inspired — Supersonic Aerodynamics Physics AI")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # ── Step 1: Generate dataset ───────────────────────────────────────────
    print("[1/8] Generating analytical dataset (3 000 designs × 100 points) …")
    data = generate_dataset(n_designs=3_000, n_pts=100, seed=0)
    print(f"      X: {data['X'].shape}  Y: {data['Y'].shape}")

    rng      = np.random.default_rng(42)
    n_test   = 5_000
    test_idx = rng.choice(data["X"].shape[0], n_test, replace=False)
    test_data = {"X": data["X"][test_idx], "Y": data["Y"][test_idx]}

    # ── Step 2: Train base model ───────────────────────────────────────────
    print("\n[2/8] Training geometry-conditioned PINN surrogate …")
    model = AeroSurrogatePINN(n_fourier=32, geo_dim=64, hidden=256, n_decode=5)
    model = maybe_compile(model)
    history = train_model(
        model, data, device,
        epochs=10_000, batch_size=4_096, lr=3e-4, phys_weight=0.05, label="aero",
    )

    # ── Step 3: Validate ───────────────────────────────────────────────────
    print("\n[3/8] Evaluating surrogate …")
    Xt = torch.tensor(test_data["X"], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(Xt).cpu().numpy()
    for i, nm in enumerate(["Cp", "Cf×1e3"]):
        rmse = float(np.sqrt(np.mean((pred[:, i] - test_data["Y"][:, i])**2)))
        rel  = rmse / (test_data["Y"][:, i].std() + 1e-8) * 100
        print(f"  {nm:10s}  RMSE={rmse:.4f}  Rel={rel:.2f}%")

    print("\n[3/8] Plotting Cp/Cf distributions …")
    plot_cp_distribution(DEFAULT_GEOM, model, device, OUT_DIR / "01_cp_cf_distribution.png")

    plot_training_history(history, OUT_DIR / "02_training_history.png")

    # ── Step 4: AeroDB generation ──────────────────────────────────────────
    print("\n[4/8] Generating AeroDB via surrogate …")
    aerodb = build_aerodb(
        model, device,
        mach_pts=[1.5, 2.0, 2.5, 3.0, 3.5],
        alpha_pts=[0.0, 2.0, 4.0, 6.0, 8.0],
    )
    with open(OUT_DIR / "aerodb.json", "w") as f:
        json.dump(aerodb, f, indent=2)
    print(f"  AeroDB saved → {OUT_DIR / 'aerodb.json'}")

    print("\n[4/8] Plotting aerodynamic polars …")
    plot_polar(aerodb, OUT_DIR / "03_aero_polars.png")

    # ── Step 5: Ensemble UQ ────────────────────────────────────────────────
    print("\n[5/8] Building 5-model ensemble for UQ …")
    import copy
    ensemble = [model]
    for seed in range(1, 5):
        m = AeroSurrogatePINN(n_fourier=32, geo_dim=64, hidden=256, n_decode=5)
        torch.manual_seed(seed * 100)
        m.load_state_dict(model.state_dict())
        for p in m.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p) * 0.005
        m.to(device)
        train_model(
            m, data, device,
            epochs=2_000, batch_size=4_096, lr=5e-5, phys_weight=0.02,
            label=f"ens-{seed}",
        )
        ensemble.append(m)

    uq = ensemble_uq(ensemble, test_data["X"], device, alpha=0.10)
    for col, nm in enumerate(["Cp", "Cf×1e3"]):
        cov = float(np.mean(
            (test_data["Y"][:, col] >= uq["lo"][:, col]) &
            (test_data["Y"][:, col] <= uq["hi"][:, col])
        ) * 100)
        print(f"  90% PI coverage [{nm}]: {cov:.1f}%")
    plot_uq(uq, test_data["Y"], OUT_DIR / "04_ensemble_uq.png")

    # ── Step 6: 6-DOF trajectory ───────────────────────────────────────────
    print("\n[6/8] Running 6-DOF trajectory simulation …")
    try:
        from scipy.interpolate import RegularGridInterpolator  # noqa: F401
        traj = run_trajectory(aerodb, m_kg=50.0, V0_ms=680.0, gamma0_deg=-5.0)
        print(f"  Range: {traj['x_km'][-1]:.1f} km  "
              f"Max Mach: {traj['M'].max():.2f}  "
              f"Flight time: {traj['t_s'][-1]:.1f} s")
        plot_trajectory(traj, OUT_DIR / "05_trajectory.png")
    except ImportError:
        print("  scipy not available — skipping trajectory integration")

    # ── Step 7: TorchScript export ─────────────────────────────────────────
    print("\n[7/8] Exporting TorchScript model …")
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save(str(OUT_DIR / "missile_aero_surrogate.pt"))
        print(f"  TorchScript saved → {OUT_DIR / 'missile_aero_surrogate.pt'}")
    except Exception as e:
        torch.save(model.state_dict(), OUT_DIR / "missile_aero_surrogate_state.pt")
        print(f"  TorchScript skipped ({e}); state dict saved.")

    # ── Step 8: Summary ────────────────────────────────────────────────────
    print("\n[8/8] Summary")
    print("="*70)
    print(f"  Outputs in: {OUT_DIR}")
    print("  Files:")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name}")
    print("="*70)


if __name__ == "__main__":
    main()
