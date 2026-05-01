"""
01_spring_mass_pinn.py — Damped Spring-Mass System with PINN Co-simulation

System
------
    m * x''(t) + c * x'(t) + k * x(t) = F(t)

    m = 1.0 kg,  k = 4.0 N/m,  c = 0.4 N·s/m  (under-damped)
    F(t) = A * sin(omega_f * t)   (harmonic forcing, A=0.5, omega_f=1.0)

    Analytical solution (homogeneous + particular) is used as ground truth.

Co-simulation graph
-------------------
    [ForceNode] --F--> [MassNode] --x,v--> (recorder)
                           ^-- x_prev, v_prev (feedback from previous step)

    ForceNode  — AnalyticalNode computing F(t)
    MassNode   — PINNNode: a small MLP that takes (x_prev, v_prev, F, t) and
                 predicts (x_next, v_next), supervised by a physics residual
                 from the ODE discretised with a trapezoidal rule.

Training loop
-------------
    For each epoch:
      1. Reset engine.
      2. Unroll N_UNROLL steps (short rollout) through the differentiable graph.
      3. Accumulate CoSimLoss (data + physics).
      4. Backpropagate and update PINN parameters.

    After training, run a full simulation and compare with the analytical solution.
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

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneaple_cosim import (
    AnalyticalNode,
    CoSimEngine,
    CoSimGraph,
    CoSimLoss,
    PINNNode,
    TrajectoryRecorder,
)

# ============================================================
# 0) Reproducibility + physical constants
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

M      = 1.0    # mass [kg]
K      = 4.0    # spring stiffness [N/m]
C      = 0.4    # damping [N·s/m]
A_F    = 0.5    # forcing amplitude [N]
W_F    = 1.0    # forcing frequency [rad/s]
DT     = 0.02   # time step [s]
T_END  = 20.0   # simulation end [s]

# ============================================================
# 1) Analytical solution  (under-damped free + particular)
# ============================================================
omega_n  = math.sqrt(K / M)               # natural freq
zeta     = C / (2 * math.sqrt(M * K))     # damping ratio  (< 1 → under-damped)
omega_d  = omega_n * math.sqrt(1 - zeta ** 2)

# Particular solution coefficients for F = A*sin(w_f*t)
denom   = (K - M * W_F**2) ** 2 + (C * W_F) ** 2
X_p     =  A_F * (K - M * W_F**2) / denom
X_pp    = -A_F * (C * W_F) / denom


def analytical_x(t: np.ndarray) -> np.ndarray:
    """Displacement x(t) starting from rest at x(0)=0."""
    x_part = X_p * np.sin(W_F * t) + X_pp * np.cos(W_F * t)
    # Initial conditions: x(0)=0, v(0)=0
    C1 = -X_pp
    C2 = (-X_p * W_F - zeta * omega_n * C1) / omega_d
    x_hom = np.exp(-zeta * omega_n * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))
    return x_hom + x_part


def analytical_v(t: np.ndarray) -> np.ndarray:
    return np.gradient(analytical_x(t), t)


t_ref = np.arange(0.0, T_END + DT, DT)
x_ref = analytical_x(t_ref)
v_ref = analytical_v(t_ref)

print(f"System: m={M}  k={K}  c={C}  omega_n={omega_n:.3f}  zeta={zeta:.3f}")
print(f"Forcing: A={A_F}  omega_f={W_F}")
print(f"Simulation: dt={DT}  T={T_END}  steps={len(t_ref)-1}")


# ============================================================
# 2) PINN model — small MLP
#    Inputs:  [x_prev, v_prev, F, t]   (4 scalars)
#    Outputs: [x_next, v_next]          (2 scalars)
# ============================================================

class SpringMassNet(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


pinn_model = SpringMassNet(hidden=64)
print(f"\nPINN parameters: {sum(p.numel() for p in pinn_model.parameters())}")


# ============================================================
# 3) Physics residual function (trapezoidal ODE discretisation)
#
#    ODE:  m*x'' + c*x' + k*x = F
#    State: (x, v) with v = x'
#    Trapezoidal:
#       x_next = x_prev + dt/2*(v_prev + v_next)
#       v_next = v_prev + dt/2*(a_prev + a_next)
#    where a = (F - c*v - k*x) / m
# ============================================================

def _accel(x: torch.Tensor, v: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return (F - C * v - K * x) / M


def spring_mass_physics(
    node: PINNNode,
    inputs: Dict[str, torch.Tensor],
    t: float,
    dt: float,
) -> torch.Tensor:
    x_prev = inputs["x_prev"]
    v_prev = inputs["v_prev"]
    F_cur  = inputs["F_ext"]

    # Forward pass to get predictions
    inp = torch.cat([x_prev, v_prev, F_cur, torch.tensor([[t]])], dim=-1)
    out = node.model(inp)
    x_next = out[:, 0:1]
    v_next = out[:, 1:2]

    # Forcing at next step
    F_next = torch.tensor([[A_F * math.sin(W_F * (t + dt))]])

    # Trapezoidal residuals
    a_prev = _accel(x_prev, v_prev, F_cur)
    a_next = _accel(x_next, v_next, F_next)

    res_x = x_next - x_prev - 0.5 * dt * (v_prev + v_next)
    res_v = v_next - v_prev - 0.5 * dt * (a_prev + a_next)

    return res_x.pow(2).mean() + res_v.pow(2).mean()


# ============================================================
# 4) Build co-simulation graph
# ============================================================

def make_force_node() -> AnalyticalNode:
    def force_fn(inputs: Dict, t: float, dt: float) -> Dict:
        F = torch.tensor([[A_F * math.sin(W_F * t)]])
        return {"F": F}
    return AnalyticalNode("forcing", force_fn, input_ports=[], output_ports=["F"])


def make_mass_node() -> PINNNode:
    return PINNNode(
        name="mass",
        model=pinn_model,
        input_ports=["x_prev", "v_prev", "F_ext"],
        output_ports=["x", "v"],
        physics_fn=spring_mass_physics,
        physics_weight=1.0,
    )


def build_graph() -> CoSimGraph:
    g = CoSimGraph()
    g.add_node(make_force_node())
    g.add_node(make_mass_node())
    g.connect("forcing.F",  "mass.F_ext")
    g.connect("mass.x",     "mass.x_prev")   # feedback
    g.connect("mass.v",     "mass.v_prev")   # feedback
    return g


# ============================================================
# 5) Training loop
# ============================================================
N_UNROLL = 10           # steps per training rollout
EPOCHS   = 300
LR       = 3e-3

graph    = build_graph()
criterion = CoSimLoss(data_weight=1.0, physics_weight=2.0, coupling_weight=0.0)
optimizer = optim.Adam(graph.trainable_parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print(f"\n=== Training  (epochs={EPOCHS}, unroll={N_UNROLL} steps) ===")

history = {"loss": [], "physics": [], "data": []}

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # Short rollout from t=0
    engine = CoSimEngine(graph, loop_solver="gauss_seidel", max_iter=1)
    engine.reset()
    engine.initialize_ports({
        "mass": {
            "x":      torch.zeros(1, 1),
            "v":      torch.zeros(1, 1),
            "x_prev": torch.zeros(1, 1),
            "v_prev": torch.zeros(1, 1),
        }
    })

    total_loss = torch.tensor(0.0)
    total_phys = 0.0
    total_data = 0.0

    for step_i in range(N_UNROLL):
        t = step_i * DT
        port_vals = engine.step(t, DT)

        # Data targets from analytical solution
        x_true = torch.tensor([[float(analytical_x(np.array([t + DT]))[0])]])
        v_true = torch.tensor([[float(analytical_v(np.array([t + DT]))[0])]])
        targets = {"mass.x": x_true, "mass.v": v_true}

        loss, info = criterion(port_vals, graph, targets=targets)
        total_loss = total_loss + loss
        total_phys += info["physics"]
        total_data += info["data"]

    total_loss = total_loss / N_UNROLL
    total_loss.backward()
    nn.utils.clip_grad_norm_(graph.trainable_parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    history["loss"].append(float(total_loss.detach()))
    history["physics"].append(total_phys / N_UNROLL)
    history["data"].append(total_data / N_UNROLL)

    if (epoch + 1) % 50 == 0:
        print(
            f"  Epoch {epoch+1:4d} | loss={history['loss'][-1]:.5f} "
            f"| physics={history['physics'][-1]:.5f} "
            f"| data={history['data'][-1]:.5f}"
        )

print("Training complete.")


# ============================================================
# 6) Full simulation with trained model
# ============================================================
print("\n=== Full simulation (T=20 s) ===")

recorder = TrajectoryRecorder()
recorder.watch_node("mass", ["x", "v"])
recorder.watch("forcing", "F")

graph_eval = build_graph()
engine_eval = CoSimEngine(graph_eval, recorder=recorder, loop_solver="gauss_seidel")
engine_eval.reset()
engine_eval.initialize_ports({
    "mass": {
        "x":      torch.zeros(1, 1),
        "v":      torch.zeros(1, 1),
        "x_prev": torch.zeros(1, 1),
        "v_prev": torch.zeros(1, 1),
    }
})

# Copy trained weights into eval graph
eval_mass_node = graph_eval.node("mass")
eval_mass_node.model.load_state_dict(pinn_model.state_dict())
eval_mass_node.model.eval()

with torch.no_grad():
    engine_eval.run(T=T_END, dt=DT)

traj_x = recorder.get("mass", "x")
traj_v = recorder.get("mass", "v")

x_pred = traj_x.values.squeeze()
v_pred = traj_v.values.squeeze()
t_sim  = traj_x.times

# Align with reference (reference starts at 0, sim records after first step)
n_sim = len(t_sim)
x_ref_aligned = x_ref[1:n_sim + 1]
v_ref_aligned = v_ref[1:n_sim + 1]

mae_x = float(np.mean(np.abs(x_pred - x_ref_aligned)))
mae_v = float(np.mean(np.abs(v_pred - v_ref_aligned)))
rmse_x = float(np.sqrt(np.mean((x_pred - x_ref_aligned) ** 2)))

print(f"  MAE  displacement x: {mae_x:.5f} m")
print(f"  MAE  velocity     v: {mae_v:.5f} m/s")
print(f"  RMSE displacement x: {rmse_x:.5f} m")
print(f"\n  Graph topology: cycles={graph_eval.has_cycles()}")
print(f"  Execution order: {graph_eval.execution_order()}")


# ============================================================
# 7) (Optional) Plots
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Displacement
    axes[0].plot(t_ref, x_ref, "k-", linewidth=1.5, label="Analytical")
    axes[0].plot(t_sim, x_pred, "r--", linewidth=1.5, alpha=0.85, label="PINN CoSim")
    axes[0].set_ylabel("x [m]")
    axes[0].set_title("Damped Spring-Mass: PINN Co-simulation vs. Analytical")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Velocity
    axes[1].plot(t_ref, v_ref, "k-", linewidth=1.5, label="Analytical")
    axes[1].plot(t_sim, v_pred, "b--", linewidth=1.5, alpha=0.85, label="PINN CoSim")
    axes[1].set_ylabel("v [m/s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Training loss
    axes[2].semilogy(history["loss"],    label="Total", linewidth=1.5)
    axes[2].semilogy(history["physics"], label="Physics", linestyle="--")
    axes[2].semilogy(history["data"],    label="Data",    linestyle="-.")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss (log scale)")
    axes[2].set_title("Training loss history")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path("outputs") / "01_spring_mass_cosim.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved: {out_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib not available -- plots skipped)")

print("\nExample complete.")
