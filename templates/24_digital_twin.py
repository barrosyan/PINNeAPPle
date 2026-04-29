"""24_digital_twin.py — Digital twin with live monitoring and data assimilation.

Demonstrates:
- DigitalTwinRuntime: wraps a PINN surrogate for real-time state estimation
- StateAssimilator: Sequential assimilation of sensor observations
- AnomalyDetector: threshold-based anomaly flagging
- SyntheticSensorStream: simulates MQTT-style sensor data without a broker

This template runs fully offline (no network required) using a synthetic sensor
stream that replays pre-computed "measurements".
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_digital_twin.runtime import DigitalTwinRuntime, DigitalTwinConfig
from pinneaple_digital_twin.assimilator import StateAssimilator, AssimilatorConfig
from pinneaple_digital_twin.anomaly import AnomalyDetector, AnomalyConfig
from pinneaple_digital_twin.sensors import SyntheticSensorStream


# ---------------------------------------------------------------------------
# Physical system: 1D transient heat equation surrogate
# Temperature field T(x, t) predicted by a PINN surrogate
# Sensors: 5 thermocouples at fixed x positions
# Anomaly: step change in boundary temperature at t=5 s
# ---------------------------------------------------------------------------

SENSOR_POSITIONS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
T_LEFT_NORMAL  = 100.0    # °C
T_LEFT_ANOMALY = 180.0    # °C  (injected at t=5)
T_RIGHT        = 20.0     # °C
ALPHA          = 0.05     # thermal diffusivity
L              = 1.0      # domain length


def T_exact(x: np.ndarray, t: float, T_left: float) -> np.ndarray:
    """Analytical solution via separation of variables (truncated)."""
    Ts = T_left + (T_right - T_left) * x / L  # steady-state
    n_terms = 20
    transient = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        bn = (2 / L) * np.trapz(
            (np.zeros_like(x)) * np.sin(n * math.pi * x / L), x
        )
        transient += bn * np.exp(-ALPHA * (n * math.pi / L)**2 * t) * \
                     np.sin(n * math.pi * x / L)
    return Ts + transient


def build_surrogate() -> nn.Module:
    """Simple surrogate mapping (x, t) → T."""
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def train_surrogate(model: nn.Module, n_epochs: int = 3000) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_all = np.linspace(0, L, 50, dtype=np.float32)
    t_all = np.linspace(0, 10, 50, dtype=np.float32)
    xx, tt = np.meshgrid(x_all, t_all)
    T_vals = T_exact(xx.ravel(), tt.ravel(), T_LEFT_NORMAL).astype(np.float32)
    xt = torch.tensor(
        np.stack([xx.ravel(), tt.ravel()], axis=1), dtype=torch.float32
    )
    T_gt = torch.tensor(T_vals[:, None])

    for _ in range(n_epochs):
        opt.zero_grad()
        (model(xt) - T_gt).pow(2).mean().backward()
        opt.step()
    print(f"  surrogate final MSE = {(model(xt) - T_gt).pow(2).mean().item():.4e}")


def main():
    torch.manual_seed(5)
    print("Training heat surrogate ...")
    surrogate = build_surrogate()
    train_surrogate(surrogate, n_epochs=3000)
    surrogate.eval()

    # --- Synthetic sensor stream -------------------------------------------
    # 200 time steps, 0.1 s apart; anomaly injected at step 50 (t=5 s)
    n_steps = 100
    dt      = 0.1
    rng     = np.random.default_rng(42)

    def gen_observations(step: int) -> dict:
        t = step * dt
        T_left = T_LEFT_ANOMALY if t >= 5.0 else T_LEFT_NORMAL
        vals = T_exact(SENSOR_POSITIONS, t, T_left) + rng.normal(0, 0.5, len(SENSOR_POSITIONS))
        return {f"sensor_{i}": float(v) for i, v in enumerate(vals)}

    stream = SyntheticSensorStream(
        observation_fn=gen_observations,
        n_steps=n_steps,
        dt=dt,
    )

    # --- Digital twin config -----------------------------------------------
    dt_config = DigitalTwinConfig(
        update_interval=dt,
        n_state_vars=len(SENSOR_POSITIONS),
        device="cpu",
    )

    # --- State assimilator -------------------------------------------------
    assim_config = AssimilatorConfig(
        method="enkf",            # ensemble Kalman filter
        n_ensemble=20,
        obs_noise_std=0.5,
        state_dim=50,             # discretised temperature field
        obs_dim=len(SENSOR_POSITIONS),
    )

    def obs_operator(state: torch.Tensor, t: float) -> torch.Tensor:
        """Map state (field) → sensor readings via surrogate."""
        x_t = torch.tensor(SENSOR_POSITIONS[:, None], dtype=torch.float32)
        t_t = torch.full((len(SENSOR_POSITIONS), 1), t, dtype=torch.float32)
        xt  = torch.cat([x_t, t_t], dim=1)
        with torch.no_grad():
            return surrogate(xt).squeeze(-1)

    assimilator = StateAssimilator(
        config=assim_config,
        observation_operator=obs_operator,
    )

    # --- Anomaly detector --------------------------------------------------
    anomaly_config = AnomalyConfig(
        method="residual_threshold",
        threshold=3.0,      # flag if innovation > 3 * obs_noise
        window=5,
    )
    detector = AnomalyDetector(config=anomaly_config)

    # --- Runtime -----------------------------------------------------------
    runtime = DigitalTwinRuntime(
        surrogate=surrogate,
        assimilator=assimilator,
        anomaly_detector=detector,
        config=dt_config,
    )

    # --- Replay loop -------------------------------------------------------
    print("\nReplaying sensor stream ...")
    timestamps, innovations, anomaly_flags, mean_T = [], [], [], []

    for step, obs_dict in enumerate(stream):
        t = step * dt
        obs_vec = np.array(list(obs_dict.values()), dtype=np.float32)
        obs_tensor = torch.tensor(obs_vec)

        state_est, innovation = runtime.update(obs_tensor, t=t)
        flag = detector.check(innovation)

        timestamps.append(t)
        innovations.append(float(innovation.abs().mean().item()))
        anomaly_flags.append(flag)
        mean_T.append(float(state_est.mean().item()))

        if flag:
            print(f"  t={t:.1f}s  ANOMALY DETECTED!  innov={innovations[-1]:.3f}")

    # --- Visualisation -----------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(timestamps, mean_T, "b-", label="Estimated mean T")
    axes[0].axvline(5.0, color="red", ls="--", label="Anomaly injected at t=5")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Digital twin state estimate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(timestamps, innovations, "k-", label="Mean |innovation|")
    an_t = [t for t, f in zip(timestamps, anomaly_flags) if f]
    an_v = [innovations[i] for i, f in enumerate(anomaly_flags) if f]
    if an_t:
        axes[1].scatter(an_t, an_v, c="red", zorder=5, s=40, label="Anomaly flag")
    axes[1].axhline(anomaly_config.threshold * 0.5, color="orange", ls="--",
                    label=f"Threshold")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Innovation magnitude")
    axes[1].set_title("Anomaly detection signal")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("24_digital_twin_result.png", dpi=120)
    print("Saved 24_digital_twin_result.png")


if __name__ == "__main__":
    main()
