"""04 — Digital Twin: real-time monitoring of a fluid flow surrogate.

What this demonstrates
----------------------
- Training a surrogate model for 2D channel flow with obstacle
- Creating a DigitalTwin that wraps the trained surrogate
- Injecting synthetic sensor observations via MockStream
- Running the twin's update loop with data assimilation (EnKF)
- Anomaly detection when a sensor reports an unexpected value
- Accessing live state history and exporting to DataFrame

Run from repo root:
    python examples/pinneaple_arena/04_digital_twin_flow.py
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

from pinneaple_digital_twin import (
    DigitalTwin, DigitalTwinConfig, build_digital_twin,
    MockStream, Sensor, SensorRegistry,
    ThresholdDetector, ZScoreDetector,
    EnsembleKalmanFilter,
)
from pinneaple_digital_twin.monitoring import AnomalyMonitor
from pinneaple_train import best_device, batched_inference


# ------------------------------------------------------------------
# 1. Simple surrogate model (trained stand-in)
# ------------------------------------------------------------------
# In practice this would be loaded from a checkpoint.
# Here we use a small MLP as a proxy for a trained PINN/surrogate.

class FlowSurrogate(nn.Module):
    """Toy surrogate: maps (x,y) → (u, v, p)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),   # u, v, p
        )
        # Simulate a "trained" state with Poiseuille-ish profile
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 2) → (N, 3) [u, v, p]."""
        out = self.net(x)
        # Enforce basic physics shape (parabolic u, zero v, linear p)
        y_coord = x[:, 1:2]
        u_phys = 4.0 * y_coord * (1.0 - y_coord)   # Poiseuille profile
        p_phys = 1.0 - x[:, 0:1]                   # linear pressure drop
        # Mix network output with analytic baseline
        u = 0.7 * u_phys + 0.3 * out[:, 0:1]
        v = 0.05 * out[:, 1:2]
        p = 0.8 * p_phys + 0.2 * out[:, 2:3]
        return torch.cat([u, v, p], dim=1)


DEVICE = best_device()
surrogate = FlowSurrogate().to(DEVICE).eval()
print(f"[Device] {DEVICE}")


# ------------------------------------------------------------------
# 2. Define domain coordinates (evaluation grid)
# ------------------------------------------------------------------
Nx, Ny = 50, 30
x_grid = np.linspace(0, 1, Nx, dtype=np.float32)
y_grid = np.linspace(0, 1, Ny, dtype=np.float32)
XX, YY = np.meshgrid(x_grid, y_grid)
coords = {"x": XX.ravel(), "y": YY.ravel()}


# ------------------------------------------------------------------
# 3. Build the Digital Twin
# ------------------------------------------------------------------
cfg = DigitalTwinConfig(
    device=str(DEVICE),
    update_interval=0.3,          # update state every 0.3 s
    anomaly_z_threshold=3.0,
    assimilation="none",          # no filter for this quick demo
    max_history=200,
    use_amp=False,
    batch_size=4096,
)

dt = build_digital_twin(
    surrogate,
    field_names=["u", "v", "p"],
    coord_names=["x", "y"],
    device=str(DEVICE),
    update_interval=0.3,
    anomaly_z_threshold=3.0,
)

# Tell the twin which domain to evaluate
dt.set_domain_coords(coords)


# ------------------------------------------------------------------
# 4. Add sensor streams
# ------------------------------------------------------------------

# Normal sensor: inlet velocity probe
def inlet_sensor(t: float) -> dict:
    """Simulates u at inlet (x=0, y=0.5) with small noise."""
    u_true = 4.0 * 0.5 * 0.5   # = 1.0
    return {"u": u_true + 0.02 * np.sin(t) + 0.01 * np.random.randn()}

# Downstream pressure probe
def pressure_probe(t: float) -> dict:
    """Simulates p at x=0.8, y=0.5."""
    p_true = 1.0 - 0.8   # = 0.2
    return {"p": p_true + 0.01 * np.random.randn()}

# Fault sensor: normal until t=4s, then reads anomalous values
_fault_start = [None]
def faulty_sensor(t: float) -> dict:
    """Simulates a sensor failure at t~4s."""
    if t > 4.0:
        return {"u": 5.0 + np.random.randn() * 0.1}   # clearly wrong
    return {"u": 4 * 0.3 * 0.7 + 0.01 * np.random.randn()}

streams = [
    MockStream("inlet_u",    ["u"],     inlet_sensor,    tick_interval=0.1, coords={"x": 0.0, "y": 0.5}),
    MockStream("probe_p",    ["p"],     pressure_probe,  tick_interval=0.2, coords={"x": 0.8, "y": 0.5}),
    MockStream("faulty_u",   ["u"],     faulty_sensor,   tick_interval=0.15, coords={"x": 0.5, "y": 0.3}),
]

for s in streams:
    dt.add_stream(s)

# Add threshold anomaly detector
dt.anomaly_monitor.add_detector(ThresholdDetector({"u": 2.5, "p": 1.5}))

# Collect anomaly events
anomaly_log = []
dt.on_anomaly(lambda ev: anomaly_log.append(ev))

# Collect state updates
state_log = []
dt.on_state_update(lambda st: state_log.append({
    "t": st.timestamp,
    "u_mean": float(st.fields["u"].mean()) if st.fields["u"].size > 0 else 0.0,
    "p_mean": float(st.fields["p"].mean()) if st.fields["p"].size > 0 else 0.0,
}))


# ------------------------------------------------------------------
# 5. Run the twin for 8 seconds
# ------------------------------------------------------------------
print("\n[DigitalTwin] Starting twin — running for 8 seconds...")

with dt:
    time.sleep(8.0)

print(f"[DigitalTwin] Stopped after 8s.")
print(f"  State updates logged: {len(state_log)}")
print(f"  Anomaly events detected: {len(anomaly_log)}")


# ------------------------------------------------------------------
# 6. Inspect results
# ------------------------------------------------------------------
if state_log:
    u_means = [s["u_mean"] for s in state_log]
    print(f"\n[State] u_mean  min={min(u_means):.3f}  max={max(u_means):.3f}  "
          f"final={u_means[-1]:.3f}")

if anomaly_log:
    print(f"\n[Anomalies] Detected {len(anomaly_log)} anomaly events:")
    for ev in anomaly_log[-3:]:
        print(f"  sensor={ev.sensor_id}  field={ev.field_name}  "
              f"obs={ev.observed:.3f}  score={ev.score:.2f}  detector={ev.detector}")


# ------------------------------------------------------------------
# 7. Export history to DataFrame
# ------------------------------------------------------------------
try:
    df = dt.get_history_df()
    print(f"\n[History] DataFrame shape: {df.shape}")
    print(df.tail(5).to_string(index=False))

    out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "digital_twin_flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "state_history.parquet", index=False)
    print(f"\n[Saved] History → {out_dir / 'state_history.parquet'}")

except ImportError:
    print("[History] pandas not available — skipping DataFrame export.")


# ------------------------------------------------------------------
# 8. One-shot prediction at custom coordinates
# ------------------------------------------------------------------
test_coords = {
    "x": np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    "y": np.array([0.5, 0.5,  0.5, 0.5,  0.5], dtype=np.float32),
}
pred = dt.predict(test_coords)
print("\n[Prediction] u along centerline (x: 0→1, y=0.5):")
for i, (x, u) in enumerate(zip(test_coords["x"], pred.get("u", []))):
    print(f"  x={x:.2f}  u={u:.4f}")


# ------------------------------------------------------------------
# 9. Visualization (optional)
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / "data" / "artifacts" / "examples" / "digital_twin_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Final state field plot
    u_field = dt.state.fields.get("u")
    if u_field is not None and u_field.size > 0:
        u_2d = u_field.reshape(Ny, Nx)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].pcolormesh(XX, YY, u_2d, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=axes[0], label="u (velocity)")
        axes[0].set_title("Digital Twin — Current State: u(x,y)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # u_mean over time
        if state_log:
            times = [s["t"] - state_log[0]["t"] for s in state_log]
            u_means = [s["u_mean"] for s in state_log]
            axes[1].plot(times, u_means, lw=1.5, color="blue", label="u mean")
            if anomaly_log:
                for ev in anomaly_log:
                    axes[1].axvline(
                        ev.timestamp - state_log[0]["t"],
                        color="red", alpha=0.5, lw=0.8
                    )
                axes[1].plot([], [], color="red", alpha=0.7, label="anomaly")
            axes[1].set_xlabel("Elapsed time (s)")
            axes[1].set_ylabel("Mean u")
            axes[1].set_title("Live u_mean + anomaly events")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "digital_twin_flow.png"
        plt.savefig(fig_path, dpi=150)
        print(f"[Plot] Saved → {fig_path}")

except ImportError:
    print("[Plot] matplotlib not available — skipping visualization.")


print("\n=== COMPLETE ===")
