# -*- coding: utf-8 -*-
"""Terramechanics for Rovers — Bekker-Wong PINN Surrogate
=======================================================

Pipeline
--------
  1. Bekker-Wong numerical solver (scipy.quad) — reference data generation
  2. Dataset sweep over (slip ratio, sinkage) space — grid + LHS
  3. Physics-Informed Neural Network (PINN) surrogate training
  4. Multi-loss: data MSE + 4 physics constraints (zero-slip, Mohr-Coulomb,
     monotonicity, torque coupling)
  5. Surrogate evaluation vs. numerical solution + traction curves
  6. TorchScript export for real-time LunCoSim / OmniLRS integration

Physical Model (Bekker-Wong, 1969 + Wong, 1978)
---------------
  σ(θ)  — normal stress distribution over contact patch
  τ(θ)  — Mohr-Coulomb shear stress with exponential shear displacement
  F_x   — drawbar pull     = R·b·∫[τ cosθ − σ sinθ] dθ
  F_z   — normal load      = R·b·∫[σ cosθ + τ sinθ] dθ
  M_y   — driving torque   = R²·b·∫ τ dθ

Soil: GRC-1 lunar regolith simulant (compacted, 5 cm depth)
Wheel: 125 mm radius, 60 mm width, 6-wheel rover @ 40 kg total mass

References
----------
  Bekker (1969) Introduction to Terrain-Vehicle Systems
  Wong (1978) Theory of Ground Vehicles
  OmniLRS terramechanics_solver.py — github.com/OmniLRS/OmniLRS
  Rashid-1 rover traction analysis (ResearchGate, 2025)
  Peiret et al. (2018) ISTVS simulation techniques
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from pinneapple_train import best_device, maybe_compile
except ImportError:
    def best_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def maybe_compile(m, **_):
        return m

try:
    from pinneapple_simulation.particle_dynamics.terramechanics import SoilParams, WheelParams
    from pinneapple_simulation.numerical_solvers.bekker_wong import BekkerWongSolver
    from pinneapple_physics.pde_environment.presets.terramechanics import TerramechanicsResiduals
    _LIB_AVAILABLE = True
except ImportError:
    _LIB_AVAILABLE = False

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SOIL & WHEEL PARAMETERS
#    Uses pinneapple_simulation.particle_dynamics.terramechanics when available
# ═══════════════════════════════════════════════════════════════════════════════

if not _LIB_AVAILABLE:
    import math
    from dataclasses import dataclass

    @dataclass
    class SoilParams:
        c: float = 1_400.0
        phi_deg: float = 30.0
        K: float = 0.018
        k_c: float = 1_370.0
        k_phi: float = 814_000.0
        n: float = 1.0
        rho: float = 1_650.0
        g: float = 1.62
        a0: float = 0.40
        a1: float = 0.15

        @property
        def phi_rad(self) -> float:
            return math.radians(self.phi_deg)

        @property
        def tan_phi(self) -> float:
            return math.tan(self.phi_rad)

    @dataclass
    class WheelParams:
        R: float = 0.125
        b: float = 0.060
        n_wheels: int = 6
        mass_rover: float = 40.0

        def weight_per_wheel(self, g: float) -> float:
            return self.mass_rover * g / self.n_wheels

SOIL = SoilParams()
WHEEL = WheelParams()
W_WHEEL = WHEEL.mass_rover * SOIL.g / WHEEL.n_wheels


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BEKKER-WONG NUMERICAL SOLVER
#    Uses pinneapple_simulation.numerical_solvers.bekker_wong when available
# ═══════════════════════════════════════════════════════════════════════════════

if not _LIB_AVAILABLE:
    from scipy.integrate import quad as _quad

    class BekkerWongSolver:
        """Fallback Bekker-Wong solver (used when pinneapple library not installed)."""

        def __init__(self, soil=None, wheel=None):
            self.soil = soil or SOIL
            self.wheel = wheel or WHEEL

        def _contact_angles(self, z, slip):
            R = self.wheel.R
            s = self.soil
            z = max(z, 1e-6)
            theta_f = math.acos(max(-1.0, min(1.0, 1.0 - z / R)))
            theta_r = 0.0
            theta_m = (s.a0 + s.a1 * slip) * theta_f
            return theta_f, theta_r, theta_m

        def sigma(self, theta, z, slip):
            theta_f, theta_r, theta_m = self._contact_angles(z, slip)
            ksn = self.soil.k_c / self.wheel.b + self.soil.k_phi
            R, n = self.wheel.R, self.soil.n
            if theta >= theta_m:
                h = R * max(math.cos(theta) - math.cos(theta_f), 0.0)
            else:
                ratio = (theta_f - theta_m) * (theta - theta_r) / max(theta_m - theta_r, 1e-9)
                h = R * max(math.cos(theta_f - ratio) - math.cos(theta_f), 0.0)
            return ksn * h ** n

        def tau(self, theta, z, slip):
            sig = self.sigma(theta, z, slip)
            theta_f, *_ = self._contact_angles(z, slip)
            R = self.wheel.R
            j = R * ((theta_f - theta) - (1.0 - slip) * (math.sin(theta_f) - math.sin(theta)))
            return (self.soil.c + sig * self.soil.tan_phi) * (1.0 - math.exp(-j / self.soil.K))

        def forces(self, slip, z):
            theta_f, theta_r, _ = self._contact_angles(z, slip)
            R, b = self.wheel.R, self.wheel.b
            lim = max(50, int(theta_f * 200))
            Fx, _ = _quad(lambda th: self.tau(th, z, slip) * math.cos(th) - self.sigma(th, z, slip) * math.sin(th), theta_r, theta_f, limit=lim)
            Fz, _ = _quad(lambda th: self.sigma(th, z, slip) * math.cos(th) + self.tau(th, z, slip) * math.sin(th), theta_r, theta_f, limit=lim)
            My, _ = _quad(lambda th: self.tau(th, z, slip), theta_r, theta_f, limit=lim)
            return R * b * Fx, R * b * Fz, R ** 2 * b * My

        def generate_dataset(self, n_slip=45, n_sink=45, slip_range=(0.0, 0.75), sink_range=(0.002, 0.058), n_lhs=500, seed=42):
            rng = np.random.default_rng(seed)
            s_g = np.linspace(*slip_range, n_slip)
            z_g = np.linspace(*sink_range, n_sink)
            SS, ZZ = np.meshgrid(s_g, z_g)
            pts = np.stack([SS.ravel(), ZZ.ravel()], axis=1)
            lhs = np.column_stack([rng.uniform(*slip_range, n_lhs), rng.uniform(*sink_range, n_lhs)])
            pts = np.vstack([pts, lhs]).astype(np.float32)
            Y_list = []
            for s, z in pts:
                try:
                    fx, fz, my = self.forces(float(s), float(z))
                except Exception:
                    fx, fz, my = 0.0, W_WHEEL, 0.0
                Y_list.append([fx, fz, my])
            return pts, np.array(Y_list, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(
    n_slip: int = 45,
    n_sinkage: int = 45,
    slip_range: tuple = (0.0, 0.75),
    sinkage_range: tuple = (0.002, 0.058),
    seed: int = 42,
) -> dict:
    """Generate Bekker-Wong dataset using BekkerWongSolver.generate_dataset()."""
    solver = BekkerWongSolver(soil=SOIL, wheel=WHEEL)
    n_lhs = n_slip * n_sinkage // 4
    pts, Y = solver.generate_dataset(
        n_slip=n_slip, n_sink=n_sinkage,
        slip_range=slip_range, sink_range=sinkage_range,
        n_lhs=n_lhs, seed=seed,
    )
    print(f"  Generated {len(pts)} Bekker-Wong evaluations ...")
    print(f"    F_x: [{Y[:,0].min():.2f}, {Y[:,0].max():.2f}] N")
    print(f"    F_z: [{Y[:,1].min():.2f}, {Y[:,1].max():.2f}] N")
    print(f"    M_y: [{Y[:,2].min():.4f}, {Y[:,2].max():.4f}] Nm")
    return {"slip": pts[:, 0], "sinkage": pts[:, 1], "Fx": Y[:, 0], "Fz": Y[:, 1], "My": Y[:, 2]}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class Normalizer:
    """Min-max scaler to [−1, +1]. Fits on numpy, transforms numpy and torch."""

    lo: np.ndarray
    hi: np.ndarray

    def fit(self, X: np.ndarray) -> "Normalizer":
        self.lo = X.min(axis=0).astype(np.float32)
        self.hi = X.max(axis=0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.where(self.hi - self.lo > 1e-12, self.hi - self.lo, 1.0)
        return (2.0 * (X - self.lo) / rng - 1.0).astype(np.float32)

    def inverse(self, Xn: np.ndarray) -> np.ndarray:
        rng = np.where(self.hi - self.lo > 1e-12, self.hi - self.lo, 1.0)
        return ((Xn + 1.0) * rng / 2.0 + self.lo).astype(np.float32)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable transform — keeps autograd graph intact."""
        lo = torch.as_tensor(self.lo, dtype=x.dtype, device=x.device)
        hi = torch.as_tensor(self.hi, dtype=x.dtype, device=x.device)
        rng = (hi - lo).clamp(min=1e-12)
        return 2.0 * (x - lo) / rng - 1.0

    def inverse_torch(self, xn: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse — [-1, 1] back to physical space."""
        lo = torch.as_tensor(self.lo, dtype=xn.dtype, device=xn.device)
        hi = torch.as_tensor(self.hi, dtype=xn.dtype, device=xn.device)
        rng = (hi - lo).clamp(min=1e-12)
        return (xn + 1.0) * rng / 2.0 + lo


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PINN MODEL — Fourier Feature ResNet
# ═══════════════════════════════════════════════════════════════════════════════

class TerraMechanicsPINN(nn.Module):
    """Surrogate PINN: (slip_norm, sinkage_norm) → (Fx_norm, Fz_norm, My_norm).

    Architecture:
      Random Fourier Features (fixed) → ResNet MLP (Tanh) → 3 outputs
    """

    def __init__(self, n_fourier: int = 20, hidden: int = 128, depth: int = 5):
        super().__init__()
        B = torch.randn(n_fourier, 2) * 3.0
        self.register_buffer("B", B)          # frozen random projection

        in_dim = 2 * n_fourier                # sin + cos encoding
        self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(depth)
        ])
        self.head = nn.Linear(hidden, 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B.T                   # (N, n_fourier)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(self._encode(x))
        for block in self.blocks:
            h = torch.tanh(h + block(h))      # residual connection
        return self.head(h)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PHYSICS RESIDUALS
# ═══════════════════════════════════════════════════════════════════════════════

if _LIB_AVAILABLE:
    def physics_residuals(model, norm_x, norm_y, device, n_phys=256):
        """Physics constraints via pinneapple_physics.TerramechanicsResiduals."""
        Fx_range = max(float(norm_y.hi[0]) - float(norm_y.lo[0]), 1e-12)
        My_range = max(float(norm_y.hi[2]) - float(norm_y.lo[2]), 1e-12)
        R_factor = WHEEL.R * Fx_range / My_range
        _res = TerramechanicsResiduals(
            c_Pa=SOIL.c, phi_deg=SOIL.phi_deg, R_m=WHEEL.R, b_m=WHEEL.b,
            n_phys=n_phys, R_factor=R_factor,
        )
        r = _res(model, norm_x, norm_y, device)
        return {"r1_zero_slip": r["r1"], "r2_mohr_coulomb": r["r2"],
                "r3_monotonicity": r["r3"], "r4_coupling": r["r4"]}
else:
    def physics_residuals(model, norm_x, norm_y, device, n_phys=256):
        """Fallback inline physics constraints (used without pinneapple library)."""
        gen = torch.Generator(device=device).manual_seed(0)
        n1 = max(n_phys // 4, 32)
        z_r1 = torch.rand(n1, 1, device=device, generator=gen) * 0.056 + 0.002
        s_r1 = torch.zeros(n1, 1, device=device)
        x_r1 = norm_x.transform_torch(torch.cat([s_r1, z_r1], dim=-1))
        pred_r1 = model(x_r1)
        fx_r1 = norm_y.inverse_torch(pred_r1)[:, 0:1]
        r1 = fx_r1.pow(2).mean()

        n2 = max(n_phys // 2, 64)
        s_r2 = torch.rand(n2, 1, device=device, generator=gen) * 0.75
        z_r2 = torch.rand(n2, 1, device=device, generator=gen) * 0.056 + 0.002
        x_r2 = norm_x.transform_torch(torch.cat([s_r2, z_r2], dim=-1))
        pred_r2 = model(x_r2)
        phy_r2 = norm_y.inverse_torch(pred_r2)
        A_contact = WHEEL.b * WHEEL.R * math.pi
        limit = SOIL.c * A_contact + phy_r2[:, 1:2].detach() * SOIL.tan_phi
        r2 = torch.relu(phy_r2[:, 0:1] - limit).pow(2).mean()

        n3 = max(n_phys // 4, 32)
        s_r3 = (torch.rand(n3, 1, device=device, generator=gen) * 0.4).detach().requires_grad_(True)
        z_r3 = torch.rand(n3, 1, device=device, generator=gen).detach() * 0.04 + 0.004
        lo_x = torch.as_tensor(norm_x.lo, dtype=torch.float32, device=device)
        hi_x = torch.as_tensor(norm_x.hi, dtype=torch.float32, device=device)
        rng_x = (hi_x - lo_x).clamp(min=1e-12)
        x_r3 = 2.0 * (torch.cat([s_r3, z_r3], dim=-1) - lo_x) / rng_x - 1.0
        pred_r3 = model(x_r3)
        grad_s = torch.autograd.grad(pred_r3[:, 0].sum(), s_r3, create_graph=True)[0]
        r3 = torch.relu(-grad_s).pow(2).mean()

        n4 = max(n_phys // 4, 32)
        s_r4 = torch.rand(n4, 1, device=device, generator=gen) * 0.75
        z_r4 = torch.rand(n4, 1, device=device, generator=gen) * 0.056 + 0.002
        x_r4 = norm_x.transform_torch(torch.cat([s_r4, z_r4], dim=-1))
        pred_r4 = model(x_r4)
        Fx_range = max(float(norm_y.hi[0]) - float(norm_y.lo[0]), 1e-12)
        My_range = max(float(norm_y.hi[2]) - float(norm_y.lo[2]), 1e-12)
        R_factor = WHEEL.R * Fx_range / My_range
        r4 = torch.relu(R_factor * pred_r4[:, 0] - pred_r4[:, 2]).pow(2).mean()

        return {"r1_zero_slip": r1, "r2_mohr_coulomb": r2, "r3_monotonicity": r3, "r4_coupling": r4}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    epochs: int = 4_000,
    lr: float = 5e-4,
    batch_size: int = 512,
    w_data: float = 1.0,
    w_r1: float = 2.0,
    w_r2: float = 0.5,
    w_r3: float = 1.0,
    w_r4: float = 0.3,
    seed: int = 42,
) -> tuple:
    torch.manual_seed(seed)
    device = best_device()
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nStep 1/4 — Generating Bekker-Wong reference dataset ...")
    raw = generate_dataset(n_slip=45, n_sinkage=45)
    X_raw = np.column_stack([raw["slip"], raw["sinkage"]]).astype(np.float32)
    Y_raw = np.column_stack([raw["Fx"], raw["Fz"], raw["My"]]).astype(np.float32)

    norm_x = Normalizer().fit(X_raw)
    norm_y = Normalizer().fit(Y_raw)
    X = norm_x.transform(X_raw)
    Y = norm_y.transform(Y_raw)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.85 * len(X))
    X_tr_t = torch.tensor(X[idx[:n_train]], device=device)
    Y_tr_t = torch.tensor(Y[idx[:n_train]], device=device)
    X_vl_t = torch.tensor(X[idx[n_train:]], device=device)
    Y_vl_t = torch.tensor(Y[idx[n_train:]], device=device)
    print(f"    Train: {len(X_tr_t):,}  |  Val: {len(X_vl_t):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nStep 2/4 — Building TerraMechanicsPINN ...")
    model = TerraMechanicsPINN(n_fourier=20, hidden=128, depth=5).to(device)
    model = maybe_compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStep 3/4 — Training {epochs} epochs ...")
    history: dict[str, list] = {k: [] for k in ["total", "data", "val", "r1", "r2", "r3", "r4"]}
    best_val = float("inf")
    best_state = None
    n = len(X_tr_t)

    for ep in range(1, epochs + 1):
        model.train()

        idx_b = torch.randperm(n, device=device)[:min(batch_size, n)]
        xb, yb = X_tr_t[idx_b], Y_tr_t[idx_b]

        optimizer.zero_grad(set_to_none=True)

        l_data = nn.functional.mse_loss(model(xb), yb)
        phys = physics_residuals(model, norm_x, norm_y, device, n_phys=256)
        l_phys = (w_r1 * phys["r1_zero_slip"]
                  + w_r2 * phys["r2_mohr_coulomb"]
                  + w_r3 * phys["r3_monotonicity"]
                  + w_r4 * phys["r4_coupling"])
        loss = w_data * l_data + l_phys

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                l_val = nn.functional.mse_loss(model(X_vl_t), Y_vl_t).item()

            history["total"].append(loss.item())
            history["data"].append(l_data.item())
            history["val"].append(l_val)
            for k in ["r1", "r2", "r3", "r4"]:
                history[k].append(phys[f"r{k[1]}_" + {
                    "1": "zero_slip", "2": "mohr_coulomb",
                    "3": "monotonicity", "4": "coupling",
                }[k[1]]].item())

            if l_val < best_val:
                best_val = l_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if ep % 500 == 0:
                print(f"    ep={ep:5d} | total={loss.item():.3e} "
                      f"| data={l_data.item():.3e} | val={l_val:.3e} "
                      f"| r1={phys['r1_zero_slip'].item():.3e}")

    model.load_state_dict(best_state)
    print(f"\n    Best val MSE: {best_val:.4e}")
    return model, norm_x, norm_y, history, raw


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "text.color": "white",
        "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    })


def evaluate_and_plot(
    model: TerraMechanicsPINN,
    norm_x: Normalizer,
    norm_y: Normalizer,
    history: dict,
    raw: dict,
):
    print("\nStep 4/4 — Evaluating and visualizing ...")
    model.eval()
    device = next(model.parameters()).device
    solver = BekkerWongSolver()
    _setup_dark_theme()

    def _predict(slips: np.ndarray, z_val: float) -> np.ndarray:
        X_q = np.column_stack([slips, np.full_like(slips, z_val)]).astype(np.float32)
        X_qn = norm_x.transform(X_q)
        with torch.no_grad():
            Y_qn = model(torch.tensor(X_qn, device=device)).cpu().numpy()
        return norm_y.inverse(Y_qn)

    # ── Figure 1: Traction curves (PINN vs Bekker-Wong) ───────────────────────
    slips_eval = np.linspace(0.0, 0.74, 80)
    sinkage_levels = [0.005, 0.015, 0.030, 0.050]
    palette = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(
        "Terramechanics PINN — Bekker-Wong Surrogate  |  Lunar Rover (GRC-1 Regolith)",
        color="white", fontsize=13, y=1.01,
    )

    for z_eval, color in zip(sinkage_levels, palette):
        Fx_ref, Fz_ref, My_ref = zip(*[solver.forces(s, z_eval) for s in slips_eval])
        Y_pinn = _predict(slips_eval, z_eval)
        lbl = f"z={z_eval * 100:.1f} cm"

        axes1[0].plot(slips_eval, Fx_ref, "-",  color=color, lw=1.8, label=f"{lbl} ref")
        axes1[0].plot(slips_eval, Y_pinn[:, 0], "--", color=color, lw=1.3, alpha=0.85, label=f"{lbl} PINN")
        axes1[1].plot(slips_eval, Fz_ref, "-",  color=color, lw=1.8)
        axes1[1].plot(slips_eval, Y_pinn[:, 1], "--", color=color, lw=1.3, alpha=0.85)
        axes1[2].plot(slips_eval, My_ref, "-",  color=color, lw=1.8)
        axes1[2].plot(slips_eval, Y_pinn[:, 2], "--", color=color, lw=1.3, alpha=0.85)

    for ax, title, ylabel in zip(axes1,
        ["Drawbar Pull  F_x [N]", "Normal Force  F_z [N]", "Driving Torque  M_y [N·m]"],
        ["F_x [N]", "F_z [N]", "M_y [N·m]"],
    ):
        ax.set_xlabel("Slip ratio  s  [—]", color="#8b949e")
        ax.set_ylabel(ylabel, color="#8b949e")
        ax.set_title(title, color="white", pad=8)
        ax.grid(True, alpha=0.3)
    axes1[0].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out1 = OUT_DIR / "01_traction_curves.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out1}")

    # ── Figure 2: 2D force maps over (slip, sinkage) space ────────────────────
    ns = 60
    s_g = np.linspace(0.0, 0.74, ns)
    z_g = np.linspace(0.003, 0.055, ns)
    SS, ZZ = np.meshgrid(s_g, z_g)
    X_2d = np.column_stack([SS.ravel(), ZZ.ravel()]).astype(np.float32)
    with torch.no_grad():
        Y_2d = norm_y.inverse(
            model(torch.tensor(norm_x.transform(X_2d), device=device)).cpu().numpy()
        )
    Fx_map = Y_2d[:, 0].reshape(ns, ns)
    Fz_map = Y_2d[:, 1].reshape(ns, ns)
    eta_T  = np.clip(Fx_map / np.maximum(Fz_map, 1.0), -0.05, 1.1)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(
        "Terramechanics PINN — Force Maps over (Slip, Sinkage) Space",
        color="white", fontsize=13, y=1.01,
    )
    for ax, data, label, cmap in zip(
        axes2,
        [Fx_map, Fz_map, eta_T],
        ["F_x [N]", "F_z [N]", "η_T = F_x / F_z  [—]"],
        ["RdYlGn", "Blues", "RdYlBu_r"],
    ):
        im = ax.pcolormesh(SS * 100, ZZ * 100, data, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_xlabel("Slip ratio  s  [%]", color="#8b949e")
        ax.set_ylabel("Sinkage  z  [cm]", color="#8b949e")
        ax.set_title(label, color="white", pad=8)

    plt.tight_layout()
    out2 = OUT_DIR / "02_force_maps.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out2}")

    # ── Figure 3: Training history ─────────────────────────────────────────────
    steps = list(range(len(history["total"])))
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("TerraMechanicsPINN — Training Convergence", color="white", fontsize=13)

    ax_l = axes3[0]
    ax_l.semilogy(steps, history["total"], color="#58a6ff", lw=1.6, label="total")
    ax_l.semilogy(steps, history["data"],  color="#3fb950", lw=1.3, ls="--", label="data MSE")
    ax_l.semilogy(steps, history["val"],   color="#d29922", lw=1.3, ls="-.", label="val MSE")
    ax_l.set_xlabel("Epoch ×50", color="#8b949e")
    ax_l.set_ylabel("Loss", color="#8b949e")
    ax_l.set_title("Total + Data Losses", color="white")
    ax_l.legend(fontsize=9)
    ax_l.grid(True, alpha=0.3)

    ax_r = axes3[1]
    ax_r.semilogy(steps, history["r1"], color="#79c0ff", lw=1.3, label="R1: zero-slip BC")
    ax_r.semilogy(steps, history["r2"], color="#56d364", lw=1.3, label="R2: Mohr-Coulomb")
    ax_r.semilogy(steps, history["r3"], color="#e3b341", lw=1.3, label="R3: monotonicity ∂F_x/∂s")
    ax_r.semilogy(steps, history["r4"], color="#f85149", lw=1.3, label="R4: torque coupling")
    ax_r.set_xlabel("Epoch ×50", color="#8b949e")
    ax_r.set_ylabel("Physics Residual", color="#8b949e")
    ax_r.set_title("Physics Constraints Convergence", color="white")
    ax_r.legend(fontsize=9)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = OUT_DIR / "03_training_history.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out3}")

    # ── Relative L2 error report ───────────────────────────────────────────────
    print("\n    Relative L2 errors vs Bekker-Wong reference:")
    X_all = np.column_stack([raw["slip"], raw["sinkage"]]).astype(np.float32)
    Y_all = np.column_stack([raw["Fx"], raw["Fz"], raw["My"]]).astype(np.float32)
    with torch.no_grad():
        Y_pred = norm_y.inverse(
            model(torch.tensor(norm_x.transform(X_all), device=device)).cpu().numpy()
        )
    for i, name in enumerate(["F_x", "F_z", "M_y"]):
        t, p = Y_all[:, i], Y_pred[:, i]
        rel_l2 = np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12)
        print(f"      {name}: {rel_l2:.4e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EXPORT FOR LUNCO SIM
# ═══════════════════════════════════════════════════════════════════════════════

def export_surrogate(model: TerraMechanicsPINN, norm_x: Normalizer, norm_y: Normalizer):
    """Export trained surrogate as TorchScript + JSON metadata for LunCoSim."""
    model.eval()
    device = next(model.parameters()).device

    dummy = torch.zeros(1, 2, device=device)
    traced = torch.jit.trace(model, dummy)
    ts_path = OUT_DIR / "terramechanics_surrogate.pt"
    traced.save(str(ts_path))
    print(f"\n  TorchScript -> {ts_path}")

    meta = {
        "model": "TerraMechanicsPINN — Bekker-Wong Lunar Surrogate",
        "inputs": ["slip_ratio [-]", "sinkage_m [m]"],
        "outputs": ["Fx_N [N]", "Fz_N [N]", "My_Nm [N·m]"],
        "norm_x_lo": norm_x.lo.tolist(),
        "norm_x_hi": norm_x.hi.tolist(),
        "norm_y_lo": norm_y.lo.tolist(),
        "norm_y_hi": norm_y.hi.tolist(),
        "soil_params": {
            "c_Pa": SOIL.c, "phi_deg": SOIL.phi_deg, "K_m": SOIL.K,
            "k_c": SOIL.k_c, "k_phi": SOIL.k_phi, "n": SOIL.n,
            "rho_kg_m3": SOIL.rho, "g_lunar_m_s2": SOIL.g,
        },
        "wheel_params": {
            "R_m": WHEEL.R, "b_m": WHEEL.b,
            "n_wheels": WHEEL.n_wheels, "mass_rover_kg": WHEEL.mass_rover,
        },
        "usage_example": (
            "import torch, json\n"
            "ts_model = torch.jit.load('terramechanics_surrogate.pt')\n"
            "meta = json.load(open('surrogate_metadata.json'))\n"
            "lo_x = torch.tensor(meta['norm_x_lo'])\n"
            "hi_x = torch.tensor(meta['norm_x_hi'])\n"
            "lo_y = torch.tensor(meta['norm_y_lo'])\n"
            "hi_y = torch.tensor(meta['norm_y_hi'])\n"
            "x = torch.tensor([[slip, sinkage_m]])\n"
            "x_n = 2*(x-lo_x)/(hi_x-lo_x).clamp(1e-12) - 1\n"
            "with torch.no_grad(): y_n = ts_model(x_n)\n"
            "y = (y_n+1)*(hi_y-lo_y)/2 + lo_y\n"
            "Fx, Fz, My = y[0,0].item(), y[0,1].item(), y[0,2].item()"
        ),
    }
    meta_path = OUT_DIR / "surrogate_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata     -> {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  Terramechanics for Rovers — Bekker-Wong PINN Surrogate")
    print("  PINNeAPPle use case  |  Lunar regolith GRC-1  |  6-wheel rover")
    print("=" * 64)
    print(f"\n  Soil: c={SOIL.c} Pa  phi={SOIL.phi_deg} deg  K={SOIL.K} m")
    print(f"  Wheel: R={WHEEL.R} m  b={WHEEL.b} m  W/wheel={W_WHEEL:.2f} N")

    model, norm_x, norm_y, history, raw = train(
        epochs=4_000,
        lr=5e-4,
        batch_size=512,
        w_data=1.0,
        w_r1=2.0,
        w_r2=0.5,
        w_r3=1.0,
        w_r4=0.3,
    )

    evaluate_and_plot(model, norm_x, norm_y, history, raw)
    export_surrogate(model, norm_x, norm_y)

    print(f"\n  All outputs in: {OUT_DIR}")
    print("=" * 64)
