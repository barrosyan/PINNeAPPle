from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import mse, rel_l2, plot_series


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GRUForecaster(nn.Module):
    def __init__(self, d_in: int, d_h: int = 64, d_out: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=d_h, batch_first=True)
        self.head = nn.Linear(d_h, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        h, _ = self.gru(x)
        y = self.head(h[:, -1])
        return y


def _make_oscillator(T: int, dt: float, zeta: float, w0: float, seed: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    v = np.zeros(T)
    x[0] = 1.0
    v[0] = 0.0
    for k in range(1, T):
        # x'' + 2 zeta w0 x' + w0^2 x = 0
        a = -2.0 * zeta * w0 * v[k - 1] - (w0**2) * x[k - 1]
        v[k] = v[k - 1] + dt * a
        x[k] = x[k - 1] + dt * v[k]
    y = x + noise * rng.standard_normal(T)
    return x, y


def run_physics_ts_fusion(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical D — Physics + Time Series Fusion

    Demo:
      - Generate noisy observations from a damped oscillator
      - Train 2 forecasters:
          (1) purely supervised
          (2) supervised + physics constraint (discrete ODE residual)

    Output:
      - metrics and plots
    """
    prefer_cuda = bool(cfg.get("prefer_cuda", True))
    dev = _device(prefer_cuda)

    T = int(cfg.get("T", 1200))
    dt = float(cfg.get("dt", 0.01))
    zeta = float(cfg.get("zeta", 0.05))
    w0 = float(cfg.get("w0", 2.0 * np.pi))
    noise = float(cfg.get("noise_std", 0.02))
    seed = int(cfg.get("seed", 0))

    lookback = int(cfg.get("lookback", 32))
    train_steps = int(cfg.get("train_steps", 1200))
    batch = int(cfg.get("batch_size", 128))
    lr = float(cfg.get("lr", 2e-3))

    w_phys = float(cfg.get("w_phys", 0.5))

    x_true, y_obs = _make_oscillator(T=T, dt=dt, zeta=zeta, w0=w0, seed=seed, noise=noise)

    # Make supervised dataset: past window -> next value
    X = []
    Y = []
    for i in range(lookback, T - 1):
        X.append(y_obs[i - lookback:i])
        Y.append(y_obs[i + 1])
    X = np.stack(X, axis=0)[:, :, None]  # (N,L,1)
    Y = np.asarray(Y)[:, None]           # (N,1)

    N = X.shape[0]
    split = int(0.8 * N)
    Xtr, Ytr = X[:split], Y[:split]
    Xte, Yte = X[split:], Y[split:]

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    Yte_t = torch.tensor(Yte, dtype=torch.float32, device=dev)

    def train_one(use_physics: bool) -> Tuple[np.ndarray, Dict[str, float]]:
        m = GRUForecaster(d_in=1, d_h=int(cfg.get("hidden", 64)), d_out=1).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-6)
        m.train()

        for it in range(train_steps):
            idx = torch.randint(0, Xtr_t.shape[0], (batch,), device=dev)
            xb = Xtr_t[idx]
            yb = Ytr_t[idx]
            yp = m(xb)
            loss = torch.mean((yp - yb) ** 2)

            if use_physics:
                # Physics residual on predicted sequence using finite differences (approx)
                # Interpret last two points as x_{t-1}, x_t and predicted x_{t+1}.
                x_tm1 = xb[:, -2, 0:1]
                x_t = xb[:, -1, 0:1]
                x_tp1 = yp

                v_t = (x_t - x_tm1) / dt
                v_tp1 = (x_tp1 - x_t) / dt
                a_t = (v_tp1 - v_t) / dt

                # residual: a + 2 zeta w0 v + w0^2 x = 0
                res = a_t + 2.0 * zeta * w0 * v_t + (w0**2) * x_t
                loss = loss + w_phys * torch.mean(res**2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        m.eval()
        with torch.no_grad():
            pred = m(Xte_t).detach().cpu().numpy().reshape(-1)
        ref = Yte.reshape(-1)
        return pred, {
            "mse": mse(ref, pred),
            "rel_l2": rel_l2(ref, pred),
        }

    pred_sup, met_sup = train_one(use_physics=False)
    pred_phys, met_phys = train_one(use_physics=True)

    # Plot first 400 points of test
    ref = Yte.reshape(-1)
    show = min(400, ref.shape[0])
    plot_sup = plot_series(ref[:show], pred_sup[:show], title="Forecast: supervised")
    plot_phys = plot_series(ref[:show], pred_phys[:show], title="Forecast: physics + supervised")

    ranking = sorted(
        [
            {"model": "supervised", **met_sup},
            {"model": "physics_fused", **met_phys},
        ],
        key=lambda r: (r["rel_l2"], r["mse"]),
    )

    return {
        "task": "physics_time_series_fusion",
        "problem": "damped_oscillator",
        "ranking": ranking,
        "plots": {
            "supervised": plot_sup,
            "physics_fused": plot_phys,
        },
        "preview": {
            "ref": ref[:show].tolist(),
            "pred_supervised": pred_sup[:show].tolist(),
            "pred_physics": pred_phys[:show].tolist(),
        },
    }
