"""10 — Industrial digital twin: datacenter thermal monitoring.

Demonstrates a production-style digital twin for a datacenter:
- Problem loaded from pre-defined preset (datacenter_airflow_2d)
- Surrogate model predicts T, u, v, p over the rack room
- Multiple sensor streams (inlet temperature, rack outlet temp, airflow)
- Ensemble Kalman Filter for state estimation under noisy sensors
- Threshold + Z-score anomaly detectors with alert callbacks
- Rolling history export to Parquet for dashboards
- One-shot predictions at custom sensor locations

This pattern applies equally to:
  - industrial_furnace_thermal  (furnace temperature monitoring)
  - refractory_lining           (wall temperature gradients)
  - cpu_heatsink_thermal        (server thermal management)

Run from repo root:
    python examples/pinneaple_arena/10_datacenter_digital_twin.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from pinneaple_environment import get_preset
from pinneaple_digital_twin import (
    DigitalTwin, DigitalTwinConfig, build_digital_twin,
    MockStream, Sensor, SensorRegistry,
    ThresholdDetector, ZScoreDetector,
    EnsembleKalmanFilter, Observation,
)
from pinneaple_train import best_device


# ------------------------------------------------------------------
# 1. Load problem spec
# ------------------------------------------------------------------
spec = get_preset("datacenter_airflow_2d", n_racks=6, Q_rack=12000.0)

print("=" * 60)
print(f"Problem : {spec.problem_id}")
print(f"Fields  : {spec.fields}")
print(f"Domain  : {spec.domain_bounds}")
print(f"IT load : {spec.meta['total_IT_load_kW']:.1f} kW  |  Re={spec.meta['Re']:.0f}")
print(f"Alert   : T > {spec.meta['alert_T_max']} K  ({spec.meta['alert_T_max']-273.15:.0f}°C)")
print("=" * 60)


# ------------------------------------------------------------------
# 2. Surrogate model (stands for a trained PINN/FNO loaded from disk)
# ------------------------------------------------------------------
DEVICE = best_device()
T_supply = 291.0        # 18°C supply air
T_alert  = spec.meta["alert_T_max"]   # 45°C hot-aisle limit


class DatacenterSurrogate(nn.Module):
    """
    Toy surrogate for (x, y) → (u, v, p, T).

    In production, replace with:
        model = FourierNeuralOperator(...)
        model.load_state_dict(torch.load("dc_fno.pt"))
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 4),
        )
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        # Approximate physics: parabolic u, pressure drop, linear T rise
        u    =  2.5 * torch.ones_like(x_coord) + 0.1 * raw[:, 0:1]
        v    =  0.05 * raw[:, 1:2]
        p    =  1.0 - 0.1 * x_coord + 0.05 * raw[:, 2:3]
        # T rises from T_supply at inlet to higher values near racks
        T    = (T_supply + 20.0 * x_coord / 10.0 + 3.0 * torch.abs(raw[:, 3:4]))
        return torch.cat([u, v, p, T], dim=1)


surrogate = DatacenterSurrogate().to(DEVICE).eval()
print(f"[Surrogate] {sum(p.numel() for p in surrogate.parameters()):,} parameters on {DEVICE}")


# ------------------------------------------------------------------
# 3. Domain grid
# ------------------------------------------------------------------
x_bounds = spec.domain_bounds.get("x", (0.0, 10.0))
y_bounds  = spec.domain_bounds.get("y", (0.0, 2.0))
Nx, Ny   = 60, 20
x_grid   = np.linspace(*x_bounds, Nx, dtype=np.float32)
y_grid   = np.linspace(*y_bounds, Ny, dtype=np.float32)
XX, YY   = np.meshgrid(x_grid, y_grid)
domain_coords = {"x": XX.ravel(), "y": YY.ravel()}


# ------------------------------------------------------------------
# 4. EnKF for state estimation (T field, simplified)
# ------------------------------------------------------------------
# State = [T_mean, T_max] (2D for demo; production would use full field)
n_state = 2
n_obs   = 2   # two temperature sensors

def dynamics(x_state: np.ndarray) -> np.ndarray:
    """Slow drift model: state is approximately constant."""
    return x_state + np.array([0.0, 0.0])

def obs_operator(x_state: np.ndarray) -> np.ndarray:
    """Observe T_mean and T_max directly."""
    return x_state

enkf = EnsembleKalmanFilter(
    n_state=n_state,
    n_obs=n_obs,
    f=dynamics,
    h=obs_operator,
    Q=np.eye(n_state) * 0.1,
    R=np.eye(n_obs) * 0.5,
    n_ens=50,
    seed=0,
)
enkf.initialize(x0=np.array([T_supply + 5.0, T_supply + 15.0]))


# ------------------------------------------------------------------
# 5. Build digital twin
# ------------------------------------------------------------------
cfg = DigitalTwinConfig(
    device=str(DEVICE),
    update_interval=0.5,
    anomaly_z_threshold=3.0,
    assimilation="none",     # we run EnKF manually in the callback
    max_history=300,
    batch_size=2048,
)

dt = DigitalTwin(
    surrogate,
    field_names=["u", "v", "p", "T"],
    coord_names=["x", "y"],
    config=cfg,
)
dt.set_domain_coords(domain_coords)


# ------------------------------------------------------------------
# 6. Anomaly detectors
# ------------------------------------------------------------------
dt.anomaly_monitor.add_detector(
    ThresholdDetector(
        {"T": T_alert},              # absolute threshold
        rel_thresholds={"T": 0.5},   # 50% relative deviation
    )
)
dt.anomaly_monitor.add_detector(
    ZScoreDetector(z_threshold=3.5, window_size=50)
)


# ------------------------------------------------------------------
# 7. Sensor streams
# ------------------------------------------------------------------
rng = np.random.default_rng(42)

def cold_aisle_inlet(t: float) -> dict:
    """Cold aisle supply temperature + airflow velocity."""
    T_in = T_supply + 0.5 * np.sin(0.1 * t) + 0.2 * np.random.randn()
    u_in = 2.5 + 0.1 * np.random.randn()
    return {"T": T_in, "u": u_in}

def rack_outlet_center(t: float) -> dict:
    """Hot-aisle temperature at center of row."""
    T_hot = T_supply + 18.0 + 2.0 * np.sin(0.05 * t) + 0.5 * np.random.randn()
    return {"T": T_hot}

def rack_outlet_edge(t: float) -> dict:
    """Hot-aisle temperature at end of row — slight overtemp after 6s."""
    if t > 6.0:
        # Simulate a cooling failure: T spikes
        T_hot = T_alert + 3.0 + np.random.randn()
    else:
        T_hot = T_supply + 15.0 + np.random.randn()
    return {"T": T_hot}

streams = [
    MockStream("cold_inlet",      ["T", "u"], cold_aisle_inlet,
               tick_interval=0.2, coords={"x": 0.0, "y": 1.0}),
    MockStream("rack_center_out", ["T"],      rack_outlet_center,
               tick_interval=0.3, coords={"x": 5.0, "y": 1.5}),
    MockStream("rack_edge_out",   ["T"],      rack_outlet_edge,
               tick_interval=0.25, coords={"x": 9.5, "y": 1.5}),
]
for s in streams:
    dt.add_stream(s)


# ------------------------------------------------------------------
# 8. Callbacks
# ------------------------------------------------------------------
alert_log = []

def on_anomaly(ev):
    msg = (f"[ALERT] {ev.sensor_id}/{ev.field_name}  "
           f"obs={ev.observed:.1f} K  score={ev.score:.2f}  ({ev.detector})")
    print(msg)
    alert_log.append({"msg": msg, "t": ev.timestamp})

dt.on_anomaly(on_anomaly)

T_history = []
def on_state(state):
    T_field = state.fields.get("T")
    if T_field is not None and T_field.size > 0:
        T_history.append({
            "t": state.timestamp,
            "T_mean": float(T_field.mean()),
            "T_max":  float(T_field.max()),
            "T_min":  float(T_field.min()),
        })
        # Run EnKF with the latest T stats
        y_obs = np.array([T_field.mean(), T_field.max()], dtype=np.float64)
        enkf.step(y_obs)

dt.on_state_update(on_state)


# ------------------------------------------------------------------
# 9. Run the twin for 12 seconds
# ------------------------------------------------------------------
print(f"\n[DigitalTwin] Running for 12 seconds (simulated datacenter monitoring)...")
t_run_start = time.time()

with dt:
    for i in range(12):
        time.sleep(1.0)
        T_field = dt.state.fields.get("T")
        if T_field is not None and T_field.size > 0:
            print(f"  t={i+1:2d}s  T_mean={T_field.mean():.1f}K  "
                  f"T_max={T_field.max():.1f}K  "
                  f"alerts={len(alert_log)}", end="\r")

print()
print(f"\n[Done] {time.time()-t_run_start:.1f}s total")
print(f"  State updates: {len(T_history)}")
print(f"  Anomaly alerts: {len(alert_log)}")


# ------------------------------------------------------------------
# 10. EnKF state estimate summary
# ------------------------------------------------------------------
enkf_mean = enkf.mean
print(f"\n[EnKF] State estimate:")
print(f"  T_mean (estimated) = {enkf_mean[0]:.2f} K  ({enkf_mean[0]-273.15:.1f}°C)")
print(f"  T_max  (estimated) = {enkf_mean[1]:.2f} K  ({enkf_mean[1]-273.15:.1f}°C)")
print(f"  State covariance diagonal: {np.diag(enkf.covariance).round(3)}")


# ------------------------------------------------------------------
# 11. Specific point prediction
# ------------------------------------------------------------------
print("\n[Prediction] T profile along rack centerline (y=1.5, x: 0→10):")
x_probe = np.linspace(*x_bounds, 10, dtype=np.float32)
y_probe = np.full(10, 1.5, dtype=np.float32)
pred = dt.predict({"x": x_probe, "y": y_probe})
T_pred = pred.get("T", np.full(10, float("nan")))
for xi, Ti in zip(x_probe, T_pred):
    status = " *** ALERT" if Ti > T_alert else ""
    print(f"  x={xi:5.1f}m  T={Ti:.1f}K  ({Ti-273.15:.1f}°C){status}")


# ------------------------------------------------------------------
# 12. Save history and report
# ------------------------------------------------------------------
out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "datacenter_twin"
out_dir.mkdir(parents=True, exist_ok=True)

import json
with open(out_dir / "alerts.json", "w") as f:
    json.dump(alert_log, f, indent=2)

with open(out_dir / "enkf_state.json", "w") as f:
    json.dump({
        "T_mean_K": float(enkf_mean[0]),
        "T_max_K":  float(enkf_mean[1]),
        "covariance": enkf.covariance.tolist(),
    }, f, indent=2)

try:
    import pandas as pd
    df_history = dt.get_history_df()
    df_history.to_parquet(out_dir / "state_history.parquet", index=False)

    df_T = pd.DataFrame(T_history)
    df_T.to_parquet(out_dir / "T_history.parquet", index=False)
    print(f"\n[Saved] History → {out_dir}")
except ImportError:
    pass


# ------------------------------------------------------------------
# 13. Visualization
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Current T field
    T_field = dt.state.fields.get("T")
    if T_field is not None and T_field.size > 0:
        T_2d = T_field.reshape(Ny, Nx)
        im = axes[0].pcolormesh(XX, YY, T_2d, cmap="hot_r",
                                vmin=T_supply, vmax=T_alert + 5, shading="auto")
        plt.colorbar(im, ax=axes[0], label="T (K)")
        axes[0].contour(XX, YY, T_2d, levels=[T_alert], colors="cyan", linewidths=1.5)
        axes[0].set_title("Datacenter T field (cyan = alert limit)")
        axes[0].set_xlabel("x (m)  [along rack row]")
        axes[0].set_ylabel("y (m)  [height]")

    # T history
    if T_history:
        times = [h["t"] - T_history[0]["t"] for h in T_history]
        axes[1].plot(times, [h["T_mean"] for h in T_history], label="T_mean", lw=1.5)
        axes[1].plot(times, [h["T_max"]  for h in T_history], label="T_max",  lw=1.5, color="red")
        axes[1].axhline(T_alert, color="orange", lw=1, linestyle="--", label=f"Alert ({T_alert:.0f}K)")
        axes[1].set_xlabel("Elapsed time (s)")
        axes[1].set_ylabel("Temperature (K)")
        axes[1].set_title("Live T monitoring")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    # Centerline T profile
    axes[2].plot(x_probe, T_pred - 273.15, "b-o", lw=1.5, markersize=4, label="T (°C)")
    axes[2].axhline(T_alert - 273.15, color="red", lw=1.2, linestyle="--", label="Alert limit")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_title("T profile — rack centerline")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "datacenter_twin.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[Plot] {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available.")


print("\n=== COMPLETE ===")
print(f"  Total alerts : {len(alert_log)}")
print(f"  EnKF T_max   : {enkf_mean[1]:.1f} K  ({enkf_mean[1]-273.15:.1f}°C)")
print(f"  Artifacts    : {out_dir}")
