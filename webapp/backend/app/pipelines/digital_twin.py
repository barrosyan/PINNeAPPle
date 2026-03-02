from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .utils import mse, rel_l2, plot_series


def run_digital_twin(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical B — Digital Twin Builder

    Demo:
      - Synthetic sensor stream y_t = V_t
      - Simple physics-ish battery surrogate: V = OCV(SOC) - R_int * I
      - Online calibration of R_int to reduce prediction error.

    Returns:
      metrics + plots.
    """
    T = int(cfg.get("T", 600))
    dt = float(cfg.get("dt", 1.0))
    seed = int(cfg.get("seed", 0))
    noise = float(cfg.get("noise_std", 0.01))
    lr = float(cfg.get("lr", 5e-2))

    rng = np.random.default_rng(seed)

    # Synthetic current profile
    t = np.arange(T) * dt
    I = 2.0 * np.sin(2 * np.pi * t / 120.0) + 0.5 * np.sin(2 * np.pi * t / 35.0)
    I += 0.2 * rng.standard_normal(T)

    # True hidden parameter
    R_true = float(cfg.get("R_true", 0.08))

    # SOC dynamics (very rough)
    soc = np.zeros(T)
    soc[0] = 0.6
    for k in range(1, T):
        soc[k] = np.clip(soc[k - 1] - 0.0008 * I[k - 1], 0.0, 1.0)

    # OCV curve (smooth nonlinearity)
    ocv = 3.0 + 1.2 * soc - 0.15 * np.sin(4 * np.pi * soc)

    V_true = ocv - R_true * I
    V_meas = V_true + noise * rng.standard_normal(T)

    # Online estimation of R using SGD on squared error
    R_hat = float(cfg.get("R_init", 0.2))
    V_pred = np.zeros(T)
    R_hist = np.zeros(T)

    for k in range(T):
        V_pred[k] = ocv[k] - R_hat * I[k]
        err = V_pred[k] - V_meas[k]
        # d/dR (V_pred - V_meas)^2 = 2*err*dV_pred/dR = 2*err*(-I)
        R_hat = float(R_hat - lr * 2.0 * err * (-I[k]))
        R_hat = float(np.clip(R_hat, 0.0, 1.0))
        R_hist[k] = R_hat

    metrics = {
        "mse": mse(V_meas, V_pred),
        "rel_l2": rel_l2(V_meas, V_pred),
        "R_true": R_true,
        "R_final": float(R_hat),
    }

    plot_v = plot_series(V_meas, V_pred, title="Digital Twin: Voltage (sensor vs model)")
    plot_r = plot_series(np.full(T, R_true), R_hist, title="Online Calibration: R_int")

    return {
        "task": "digital_twin_builder",
        "metrics": metrics,
        "plots": {
            "voltage": plot_v,
            "resistance": plot_r,
        },
        "preview": {
            "t": t[::5].tolist(),
            "I": I[::5].tolist(),
            "V_meas": V_meas[::5].tolist(),
            "V_pred": V_pred[::5].tolist(),
            "R_hist": R_hist[::5].tolist(),
        },
    }
