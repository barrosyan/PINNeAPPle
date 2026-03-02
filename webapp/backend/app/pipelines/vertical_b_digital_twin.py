from __future__ import annotations

import base64
import io
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


def run_vertical_b(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical B (Digital Twin Builder)
    - Synthetic streaming sensor data for a "battery-like" system
    - Online parameter recalibration: estimate internal resistance R to match voltage
    """
    T = int(cfg.get("T", 400))
    dt = float(cfg.get("dt", 1.0))
    noise = float(cfg.get("noise", 0.01))
    seed = int(cfg.get("seed", 0))
    lr = float(cfg.get("lr", 0.05))
    window = int(cfg.get("window", 40))

    rng = np.random.default_rng(seed)

    # True parameters
    R_true = float(cfg.get("R_true", 0.08))
    Q = float(cfg.get("Q", 3.0))  # Ah-ish (scaled)
    soc = 0.8

    def ocv(s):
        return 3.0 + 1.2 * s - 0.1 * np.sin(6.0 * s)

    # streaming current profile
    I = 2.0 * np.sin(np.linspace(0, 18, T)) + 0.5 * rng.normal(size=T)
    I = np.clip(I, -3.0, 3.0)

    V = np.zeros(T)
    SOC = np.zeros(T)

    for k in range(T):
        soc = np.clip(soc - (I[k] * dt) / (3600.0 * Q), 0.0, 1.0)
        SOC[k] = soc
        V[k] = ocv(soc) - R_true * I[k] + noise * rng.normal()

    # online estimate of R by minimizing squared error on last window
    R_hat = float(cfg.get("R_init", 0.2))
    R_hist = []
    err_hist = []

    for k in range(T):
        s0 = max(0, k - window)
        Ik = I[s0:k+1]
        Vk = V[s0:k+1]
        Sock = SOC[s0:k+1]
        Vhat = ocv(Sock) - R_hat * Ik
        err = np.mean((Vhat - Vk) ** 2)
        # gradient w.r.t R_hat: d/dR mean((ocv - R I - V)^2) = 2*mean((Vhat-V)*(-I))
        grad = 2.0 * np.mean((Vhat - Vk) * (-Ik))
        R_hat = float(np.clip(R_hat - lr * grad, 0.0, 1.0))
        R_hist.append(R_hat)
        err_hist.append(err)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(R_hist, label="R_hat")
    ax.axhline(R_true, linestyle="--", label="R_true")
    ax.set_title("Online recalibration (internal resistance)")
    ax.set_xlabel("t")
    ax.set_ylabel("R")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "status": "ok",
        "metrics": {
            "R_true": R_true,
            "R_hat_final": R_hist[-1],
            "mse_voltage_windowed_last": float(err_hist[-1]),
        },
        "artifacts": {"R_trace_png_b64": img_b64},
        "series": {
            "I": I.tolist(),
            "V": V.tolist(),
            "SOC": SOC.tolist(),
            "R_hat": R_hist,
        },
    }
