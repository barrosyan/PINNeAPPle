"""
01_spring_mass_pinn.py — 2-DOF Coupled Spring-Mass System with Two PINN Models

Physical system
---------------
    Mass 1 (m₁=1.0 kg):  m₁·x₁'' + c₁·x₁' + k₁·x₁ + k_c·(x₁−x₂) = F(t)
    Mass 2 (m₂=0.5 kg):  m₂·x₂'' + c₂·x₂' + k₂·x₂ − k_c·(x₁−x₂) = 0

    F(t) = A·sin(ω·t),  x(0)=0, v(0)=0.

Two PINN models — demonstrating PINNeAPPle multi-model co-simulation
---------------------------------------------------------------------
    Mass 1 — SpringMassNet:   learns a₁(x₁, v₁, F, x₂)  [custom MLP]
    Mass 2 — VanillaPINN:     learns a₂(x₂, v₂, x₁)     [pinneapple_models]
             Both integrate with Euler-Cromer (symplectic).

Why acceleration, not (x_next, v_next)?
----------------------------------------
    Predicting the acceleration (force law) is unconditionally stable:
      • Euler-Cromer conserves energy long-term → no exponential drift
      • Error in a affects x_next only at O(dt²) ≈ 0.4 mm/step
      • Physics residual = MSE(a_pred, a_ODE) is clean and scale-uniform

Physics residual (normalised by characteristic acceleration scale)
------------------------------------------------------------------
    a_pred = (v_next − v_prev) / dt          ← recovered from model output
    a_true = F_ODE(x, v) / m                 ← from ODE
    loss   = ((a_pred − a_true) / A_scale)²

Co-simulation graph (algebraic loop → Gauss-Seidel, max_iter=1)
----------------------------------------------------------------
    [Forcing] ──F──► [Mass1] ──x1──► [Mass2]
                         ▲               │
                     x2 (staggered) ◄────┘

Reference: scipy.integrate.solve_ivp (RK45, rtol=1e-8)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneapple_cosim import (
    AnalyticalNode,
    CoSimEngine,
    CoSimGraph,
    CoSimLoss,
    PINNNode,
    TrajectoryRecorder,
)

# ============================================================
# 0) Physical constants
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

M1, K1, C1 = 1.0, 4.0, 0.4
M2, K2, C2 = 0.5, 2.0, 0.2
K_C        = 1.0
A_F, W_F   = 0.5, 1.0
DT         = 0.02
T_END      = 20.0

print(f"System: m1={M1} k1={K1} c1={C1}  |  m2={M2} k2={K2} c2={C2}  |  k_c={K_C}")
print(f"Forcing: A={A_F}  omega={W_F}  dt={DT}  T={T_END}")

# ============================================================
# 1) Reference solution (scipy RK45 or analytical fallback)
# ============================================================
try:
    from scipy.integrate import solve_ivp

    def _ode(t, y):
        x1, v1, x2, v2 = y
        F  = A_F * math.sin(W_F * t)
        a1 = (F  - C1*v1 - K1*x1 - K_C*(x1 - x2)) / M1
        a2 = (K_C*(x1 - x2) - C2*v2 - K2*x2) / M2
        return [v1, a1, v2, a2]

    t_ref = np.arange(0.0, T_END + DT, DT)
    sol   = solve_ivp(
        _ode, [0.0, T_END + DT], [0.0, 0.0, 0.0, 0.0],
        t_eval=t_ref, method="RK45", rtol=1e-8, atol=1e-10,
    )
    x1_ref, v1_ref = sol.y[0], sol.y[1]
    x2_ref, v2_ref = sol.y[2], sol.y[3]
    print("  Reference: scipy solve_ivp (RK45)")

except ImportError:
    omega_n = math.sqrt(K1 / M1)
    zeta    = C1 / (2 * math.sqrt(M1 * K1))
    omega_d = omega_n * math.sqrt(max(1 - zeta**2, 1e-9))
    denom   = (K1 - M1*W_F**2)**2 + (C1*W_F)**2
    Xp      =  A_F*(K1 - M1*W_F**2) / denom
    Xpp     = -A_F*(C1*W_F) / denom
    t_ref   = np.arange(0.0, T_END + DT, DT)

    def _x1_analytical(t):
        xp  = Xp*np.sin(W_F*t) + Xpp*np.cos(W_F*t)
        C1_ = -Xpp
        C2_ = (-Xp*W_F - zeta*omega_n*C1_) / omega_d
        return np.exp(-zeta*omega_n*t)*(C1_*np.cos(omega_d*t) + C2_*np.sin(omega_d*t)) + xp

    x1_ref = _x1_analytical(t_ref)
    v1_ref = np.gradient(x1_ref, t_ref)
    x2_ref = np.zeros_like(t_ref)
    v2_ref = np.zeros_like(t_ref)
    print("  Reference: analytical (decoupled, scipy not available)")


# ============================================================
# 2) Normalisation scales
# ============================================================

def _scale(arr: np.ndarray, fallback: float = 1.0) -> float:
    s = float(np.max(np.abs(arr)))
    return s if s > 1e-8 else fallback

_X1S = _scale(x1_ref, 0.15)
_V1S = _scale(v1_ref, 0.20)
_FS  = float(A_F)
_X2S = _scale(x2_ref, 0.07)
_V2S = _scale(v2_ref, 0.12)

_A1S = _scale(np.gradient(v1_ref, t_ref), 1.0)
_A2S = _scale(np.gradient(v2_ref, t_ref), 1.0)

print(f"\nNormalisation:")
print(f"  state:  x1={_X1S:.4f} m  v1={_V1S:.4f} m/s  F={_FS:.4f} N"
      f"  x2={_X2S:.4f} m  v2={_V2S:.4f} m/s")
print(f"  accel:  a1={_A1S:.4f} m/s²  a2={_A2S:.4f} m/s²")


# ============================================================
# 3) Mass 1 model — SpringMassNet (custom MLP)
#
#    Learns: a₁ = f(x₁, v₁, F, x₂)
#    Integrates with Euler-Cromer:
#        v₁_next = v₁ + a₁ · dt
#        x₁_next = x₁ + v₁_next · dt
# ============================================================

class SpringMassNet(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, v1 = x[:, 0:1], x[:, 1:2]
        scale = x.new_tensor([_X1S, _V1S, _FS, _X2S])
        a1 = self.net(x / scale) * _A1S
        v1_next = v1 + a1 * DT
        x1_next = x1 + v1_next * DT
        return torch.cat([x1_next, v1_next], dim=-1)


pinn1_model = SpringMassNet(hidden=128)
print(f"\nMass 1: SpringMassNet (accel+Euler-Cromer)  "
      f"params={sum(p.numel() for p in pinn1_model.parameters())}")


# ============================================================
# 4) Mass 2 model — VanillaPINN from pinneapple_models
#
#    VanillaPINN(in=3, out=1) predicts a₂.
#    _AccelWrapper applies input normalisation + Euler-Cromer.
# ============================================================

class _AccelWrapper(nn.Module):
    """Wraps a model that outputs acceleration (PINNOutput | Tensor).
    Inputs (physical): [x2_prev, v2_prev, x1_in]
    """
    def __init__(self, base: nn.Module, a_scale: float) -> None:
        super().__init__()
        self.base    = base
        self.a_scale = a_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2, v2 = x[:, 0:1], x[:, 1:2]
        scale  = x.new_tensor([_X2S, _V2S, _X1S])
        raw    = self.base(x / scale)
        a2     = (raw.y if hasattr(raw, "y") else raw) * self.a_scale
        v2_next = v2 + a2 * DT
        x2_next = x2 + v2_next * DT
        return torch.cat([x2_next, v2_next], dim=-1)


try:
    from pinneapple_models.pinns.vanilla import VanillaPINN
    _base2 = VanillaPINN(in_dim=3, out_dim=1, hidden=[128, 128, 128])
    nn.init.xavier_uniform_(list(_base2.parameters())[-2], gain=0.1)
    nn.init.zeros_(list(_base2.parameters())[-1])
    pinn2_model = _AccelWrapper(_base2, a_scale=_A2S)
    _m2_name    = "VanillaPINN+accel (pinneapple_models)"
except Exception:
    class _FallbackAccel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(),
                nn.Linear(128, 128), nn.Tanh(),
                nn.Linear(128, 1),
            )
            nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
            nn.init.zeros_(self.net[-1].bias)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x2, v2 = x[:, 0:1], x[:, 1:2]
            scale  = x.new_tensor([_X2S, _V2S, _X1S])
            a2     = self.net(x / scale) * _A2S
            v2_next = v2 + a2 * DT
            x2_next = x2 + v2_next * DT
            return torch.cat([x2_next, v2_next], dim=-1)
    pinn2_model = _FallbackAccel()
    _m2_name    = "custom accel MLP (fallback)"

print(f"Mass 2: {_m2_name}  "
      f"params={sum(p.numel() for p in pinn2_model.parameters())}")


# ============================================================
# 5) Physics residuals
#
#    From Euler-Cromer:  v_next = v + a·dt  →  a_pred = (v_next−v)/dt
#    ODE:                a_true = F_ODE(x,v)/m
#    Normalised loss:    ((a_pred − a_true) / A_scale)²
# ============================================================

def mass1_physics(node: PINNNode, inputs: Dict, t: float, dt: float) -> torch.Tensor:
    x1 = inputs["x1_prev"]
    v1 = inputs["v1_prev"]
    F  = inputs["F_ext"]
    x2 = inputs.get("x2_in", torch.zeros_like(x1))

    v1_next  = node._last_outputs["v1"]
    a1_pred  = (v1_next - v1) / dt
    a1_true  = (F - C1*v1 - K1*x1 - K_C*(x1 - x2)) / M1
    return ((a1_pred - a1_true) / _A1S).pow(2).mean()


def mass2_physics(node: PINNNode, inputs: Dict, t: float, dt: float) -> torch.Tensor:
    x2 = inputs["x2_prev"]
    v2 = inputs["v2_prev"]
    x1 = inputs.get("x1_in", torch.zeros_like(x2))

    v2_next  = node._last_outputs["v2"]
    a2_pred  = (v2_next - v2) / dt
    a2_true  = (K_C*(x1 - x2) - C2*v2 - K2*x2) / M2
    return ((a2_pred - a2_true) / _A2S).pow(2).mean()


# ============================================================
# 6) Co-simulation graph
# ============================================================

def make_graph() -> CoSimGraph:
    def force_fn(inputs, t, dt):
        return {"F": torch.tensor([[A_F * math.sin(W_F * t)]])}

    force_node = AnalyticalNode(
        "forcing", force_fn, input_ports=[], output_ports=["F"]
    )
    mass1_node = PINNNode(
        name="mass1",
        model=pinn1_model,
        input_ports=["x1_prev", "v1_prev", "F_ext", "x2_in"],
        output_ports=["x1", "v1"],
        physics_fn=mass1_physics,
        physics_weight=1.0,
    )
    mass2_node = PINNNode(
        name="mass2",
        model=pinn2_model,
        input_ports=["x2_prev", "v2_prev", "x1_in"],
        output_ports=["x2", "v2"],
        physics_fn=mass2_physics,
        physics_weight=1.0,
    )

    g = CoSimGraph()
    for n in (force_node, mass1_node, mass2_node):
        g.add_node(n)

    g.connect("forcing.F",  "mass1.F_ext")
    g.connect("mass1.x1",   "mass1.x1_prev")
    g.connect("mass1.v1",   "mass1.v1_prev")
    g.connect("mass1.x1",   "mass2.x1_in")
    g.connect("mass2.x2",   "mass1.x2_in")
    g.connect("mass2.x2",   "mass2.x2_prev")
    g.connect("mass2.v2",   "mass2.v2_prev")
    return g


_INIT_PORTS = {
    "mass1": {
        "x1": torch.zeros(1,1), "v1": torch.zeros(1,1),
        "x1_prev": torch.zeros(1,1), "v1_prev": torch.zeros(1,1),
        "x2_in": torch.zeros(1,1),
    },
    "mass2": {
        "x2": torch.zeros(1,1), "v2": torch.zeros(1,1),
        "x2_prev": torch.zeros(1,1), "v2_prev": torch.zeros(1,1),
        "x1_in": torch.zeros(1,1),
    },
}


# ============================================================
# 7) Training — scheduled sampling
#
#    p_ref decays 1.0 → P_REF_MIN:
#      high p_ref: reference inputs  → learns correct force law quickly
#      low  p_ref: own predictions   → eliminates exposure bias
#
#    Acceleration+Euler-Cromer means each step's gradient is
#    independent (inputs always detached) — no BPTT instability.
# ============================================================

N_UNROLL  = 50
EPOCHS    = 600
LR        = 8e-4
P_REF_MIN = 0.1

graph     = make_graph()
criterion = CoSimLoss(data_weight=1.0, physics_weight=0.5, coupling_weight=0.0)
optimizer = optim.Adam(graph.trainable_parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-6)

n_ref = len(t_ref)


def _t(arr: np.ndarray, i: int) -> torch.Tensor:
    return torch.tensor([[float(arr[i])]])


print(f"\n=== Training ({EPOCHS} epochs, window={N_UNROLL}, p_ref 1.0→{P_REF_MIN}) ===")
print(f"  Graph cycles:    {graph.has_cycles()}")
print(f"  Execution order: {graph.execution_order()}")

history: Dict = {"loss": [], "physics": [], "data": [], "p_ref": []}

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    graph.reset_all()

    p_ref = 1.0 - (1.0 - P_REF_MIN) * epoch / (EPOCHS - 1)

    max_start = n_ref - N_UNROLL - 2
    start     = np.random.randint(0, max_start)

    x1_cur = _t(x1_ref, start)
    v1_cur = _t(v1_ref, start)
    x2_cur = _t(x2_ref, start)
    v2_cur = _t(v2_ref, start)

    total_loss = torch.tensor(0.0)
    total_phys = 0.0
    total_data = 0.0

    for k in range(N_UNROLL):
        idx = start + k
        t   = float(t_ref[idx])
        F_c = torch.tensor([[A_F * math.sin(W_F * t)]])

        if np.random.random() < p_ref:
            x1_p, v1_p = _t(x1_ref, idx), _t(v1_ref, idx)
            x2_p, v2_p = _t(x2_ref, idx), _t(v2_ref, idx)
        else:
            x1_p, v1_p = x1_cur.detach(), v1_cur.detach()
            x2_p, v2_p = x2_cur.detach(), v2_cur.detach()

        out1 = graph.node("mass1").step(
            {"x1_prev": x1_p, "v1_prev": v1_p, "F_ext": F_c, "x2_in": x2_p}, t, DT
        )
        out2 = graph.node("mass2").step(
            {"x2_prev": x2_p, "v2_prev": v2_p, "x1_in": x1_p}, t, DT
        )

        x1_cur = out1["x1"].detach()
        v1_cur = out1["v1"].detach()
        x2_cur = out2["x2"].detach()
        v2_cur = out2["v2"].detach()

        targets = {
            "mass1.x1": _t(x1_ref, idx + 1),
            "mass1.v1": _t(v1_ref, idx + 1),
            "mass2.x2": _t(x2_ref, idx + 1),
            "mass2.v2": _t(v2_ref, idx + 1),
        }
        loss, info = criterion(
            {"mass1": out1, "mass2": out2, "forcing": {"F": F_c}},
            graph, targets=targets,
        )
        total_loss  = total_loss + loss
        total_phys += info.get("physics", 0.0)
        total_data += info.get("data",    0.0)

    (total_loss / N_UNROLL).backward()
    nn.utils.clip_grad_norm_(graph.trainable_parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    history["loss"].append(float(total_loss.detach() / N_UNROLL))
    history["physics"].append(total_phys / N_UNROLL)
    history["data"].append(total_data / N_UNROLL)
    history["p_ref"].append(p_ref)

    if (epoch + 1) % 100 == 0:
        print(
            f"  Epoch {epoch+1:4d}/{EPOCHS} | p_ref={p_ref:.2f}"
            f" | loss={history['loss'][-1]:.6f}"
            f" | phys={history['physics'][-1]:.6f}"
            f" | data={history['data'][-1]:.6f}"
        )

print("Training complete.")


# ============================================================
# 8) Full simulation with trained models
# ============================================================
print(f"\n=== Full simulation (T={T_END} s) ===")

recorder   = TrajectoryRecorder()
recorder.watch_node("mass1", ["x1", "v1"])
recorder.watch_node("mass2", ["x2", "v2"])
recorder.watch("forcing", "F")

graph_eval = make_graph()
graph_eval.node("mass1").model.load_state_dict(pinn1_model.state_dict())
graph_eval.node("mass2").model.load_state_dict(pinn2_model.state_dict())
graph_eval.node("mass1").model.eval()
graph_eval.node("mass2").model.eval()

engine_eval = CoSimEngine(
    graph_eval, recorder=recorder, loop_solver="gauss_seidel", max_iter=1
)
engine_eval.reset()
engine_eval.initialize_ports(_INIT_PORTS)

with torch.no_grad():
    engine_eval.run(T=T_END, dt=DT)

traj_x1 = recorder.get("mass1", "x1")
traj_v1 = recorder.get("mass1", "v1")
traj_x2 = recorder.get("mass2", "x2")
traj_v2 = recorder.get("mass2", "v2")

x1_pred = traj_x1.values.squeeze()
v1_pred = traj_v1.values.squeeze()
x2_pred = traj_x2.values.squeeze()
v2_pred = traj_v2.values.squeeze()
t_sim   = traj_x1.times

n_sim = len(t_sim)
x1_r, v1_r = x1_ref[:n_sim], v1_ref[:n_sim]
x2_r, v2_r = x2_ref[:n_sim], v2_ref[:n_sim]

print(f"\n  Mass 1 (SpringMassNet):")
print(f"    MAE  x1 = {float(np.mean(np.abs(x1_pred - x1_r))):.5f} m")
print(f"    MAE  v1 = {float(np.mean(np.abs(v1_pred - v1_r))):.5f} m/s")
print(f"    RMSE x1 = {float(np.sqrt(np.mean((x1_pred - x1_r)**2))):.5f} m")

print(f"\n  Mass 2 ({_m2_name}):")
print(f"    MAE  x2 = {float(np.mean(np.abs(x2_pred - x2_r))):.5f} m")
print(f"    MAE  v2 = {float(np.mean(np.abs(v2_pred - v2_r))):.5f} m/s")
print(f"    RMSE x2 = {float(np.sqrt(np.mean((x2_pred - x2_r)**2))):.5f} m")


# ============================================================
# 9) Plots
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # --- Mass 1 ---
    axes[0, 0].plot(t_ref, x1_ref,  "k-",  lw=1.5, label="Reference")
    axes[0, 0].plot(t_sim, x1_pred, "r--", lw=1.5, alpha=0.85, label="SpringMassNet")
    axes[0, 0].set_ylabel("x₁ [m]");  axes[0, 0].set_title("Mass 1 — displacement")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(t_ref, v1_ref,  "k-",  lw=1.5, label="Reference")
    axes[1, 0].plot(t_sim, v1_pred, "b--", lw=1.5, alpha=0.85, label="SpringMassNet")
    axes[1, 0].set_ylabel("v₁ [m/s]"); axes[1, 0].set_title("Mass 1 — velocity")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # --- Mass 2 ---
    axes[0, 1].plot(t_ref, x2_ref,  "k-",  lw=1.5, label="Reference")
    axes[0, 1].plot(t_sim, x2_pred, "g--", lw=1.5, alpha=0.85, label=_m2_name)
    axes[0, 1].set_ylabel("x₂ [m]");  axes[0, 1].set_title("Mass 2 — displacement")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(t_ref, v2_ref,  "k-",  lw=1.5, label="Reference")
    axes[1, 1].plot(t_sim, v2_pred, "m--", lw=1.5, alpha=0.85, label=_m2_name)
    axes[1, 1].set_ylabel("v₂ [m/s]"); axes[1, 1].set_title("Mass 2 — velocity")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    for ax in axes[:2, :].flat:
        ax.set_xlabel("t [s]")

    # --- Training loss ---
    ax_l = axes[2, 0]
    ax_l.semilogy(history["loss"],    lw=1.5, label="Total")
    ax_l.semilogy(history["physics"], lw=1,   ls="--", label="Physics (norm.)")
    ax_l.semilogy(history["data"],    lw=1,   ls="-.", label="Data")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss (log)")
    ax_l.set_title("Training loss (scheduled sampling)")
    ax_l.legend(fontsize=8); ax_l.grid(True, alpha=0.3)
    ax_r = ax_l.twinx()
    ax_r.plot(history["p_ref"], color="gray", lw=1, ls=":", alpha=0.7)
    ax_r.set_ylabel("p_ref", color="gray", fontsize=8)
    ax_r.tick_params(axis="y", labelcolor="gray")

    # --- Phase portrait ---
    axes[2, 1].plot(x1_ref, v1_ref,   "k-",  lw=1,   label="Mass1 ref")
    axes[2, 1].plot(x1_pred, v1_pred, "r--", lw=1,   alpha=0.8, label="Mass1 pred")
    axes[2, 1].plot(x2_ref, v2_ref,   "k:",  lw=1,   label="Mass2 ref")
    axes[2, 1].plot(x2_pred, v2_pred, "g--", lw=1,   alpha=0.8, label="Mass2 pred")
    axes[2, 1].set_xlabel("x [m]"); axes[2, 1].set_ylabel("v [m/s]")
    axes[2, 1].set_title("Phase portrait")
    axes[2, 1].legend(fontsize=8); axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle("2-DOF Coupled Spring-Mass: PINN Co-simulation", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path("outputs") / "01_coupled_spring_mass_cosim.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved: {out_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib not available — plots skipped)")

print("\nExample complete.")
