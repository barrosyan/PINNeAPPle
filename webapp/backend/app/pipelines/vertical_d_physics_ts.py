from __future__ import annotations

import base64
import io
from typing import Any, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def _simulate_oscillator(T: int, dt: float, zeta: float, w0: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros(T, dtype=np.float32)
    v = np.zeros(T, dtype=np.float32)
    # initial
    x[0] = 1.0
    v[0] = 0.0
    for k in range(T - 1):
        # x'' + 2 zeta w0 x' + w0^2 x = 0
        a = -2.0 * zeta * w0 * v[k] - (w0 * w0) * x[k]
        v[k + 1] = v[k] + dt * a
        x[k + 1] = x[k] + dt * v[k + 1]
    y = x + 0.02 * rng.normal(size=T).astype(np.float32)
    return x, y


def run_vertical_d(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical D (Physics + Time Series Fusion)
    Forecasting with and without a physics penalty (discrete residual).
    """
    T = int(cfg.get("T", 400))
    dt = float(cfg.get("dt", 0.02))
    zeta = float(cfg.get("zeta", 0.06))
    w0 = float(cfg.get("w0", 6.0))
    horizon = int(cfg.get("horizon", 40))
    window = int(cfg.get("window", 40))
    epochs = int(cfg.get("epochs", 15))
    lr = float(cfg.get("lr", 1e-3))
    lam_phys = float(cfg.get("lam_phys", 1.0))
    seed = int(cfg.get("seed", 0))

    x_true, y_obs = _simulate_oscillator(T, dt, zeta, w0, seed=seed)

    # build supervised windows: past window -> next value
    X = []
    Y = []
    for t in range(window, T - horizon):
        X.append(y_obs[t - window:t])
        Y.append(y_obs[t:t + horizon])
    X = torch.tensor(np.stack(X), dtype=torch.float32)  # (N,window)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)  # (N,horizon)

    # split
    n = X.shape[0]
    n_train = int(0.8 * n)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xte, Yte = X[n_train:], Y[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", True)) else "cpu")

    def make_gru():
        return torch.nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)

    def make_head():
        return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, horizon))

    gru_sup = make_gru().to(device)
    head_sup = make_head().to(device)
    opt_sup = torch.optim.Adam(list(gru_sup.parameters()) + list(head_sup.parameters()), lr=lr)

    gru_phys = make_gru().to(device)
    head_phys = make_head().to(device)
    opt_phys = torch.optim.Adam(list(gru_phys.parameters()) + list(head_phys.parameters()), lr=lr)

    def forward(gru, head, xb):
        xb = xb.unsqueeze(-1)  # (B,window,1)
        h, _ = gru(xb)
        feat = h[:, -1, :]
        return head(feat)

    def phys_penalty(pred_seq, dt, zeta, w0):
        # pred_seq: (B,horizon)
        # discrete residual using second difference
        x = pred_seq
        # pad with first point repeated for simplicity
        x0 = x[:, :1]
        x1 = x[:, 1:2]
        x_pad = torch.cat([x0, x, x[:, -1:]], dim=1)  # (B,h+2)
        # second derivative approx
        x_tt = (x_pad[:, 2:] - 2 * x_pad[:, 1:-1] + x_pad[:, :-2]) / (dt * dt)
        x_t = (x_pad[:, 2:] - x_pad[:, :-2]) / (2 * dt)
        res = x_tt + 2.0 * zeta * w0 * x_t + (w0 * w0) * x
        return torch.mean(res ** 2)

    Xtr = Xtr.to(device); Ytr = Ytr.to(device)
    Xte = Xte.to(device); Yte = Yte.to(device)

    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            j = perm[i:i+bs]
            xb, yb = Xtr[j], Ytr[j]

            # supervised
            pred = forward(gru_sup, head_sup, xb)
            loss = torch.mean((pred - yb) ** 2)
            opt_sup.zero_grad(); loss.backward(); opt_sup.step()

            # physics fused
            predp = forward(gru_phys, head_phys, xb)
            loss_sup = torch.mean((predp - yb) ** 2)
            loss_phys = phys_penalty(predp, dt, zeta, w0)
            loss2 = loss_sup + lam_phys * loss_phys
            opt_phys.zero_grad(); loss2.backward(); opt_phys.step()

    with torch.no_grad():
        pred_sup = forward(gru_sup, head_sup, Xte)
        pred_phys = forward(gru_phys, head_phys, Xte)
        mse_sup = torch.mean((pred_sup - Yte) ** 2).item()
        mse_phys = torch.mean((pred_phys - Yte) ** 2).item()

    # plot one example
    k = 0
    t0 = window + n_train + k
    history = y_obs[t0-window:t0]
    true_future = y_obs[t0:t0+horizon]

    ps = pred_sup[k].detach().cpu().numpy()
    pp = pred_phys[k].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(-window, 0), history, label="history")
    ax.plot(range(0, horizon), true_future, label="true")
    ax.plot(range(0, horizon), ps, label="supervised")
    ax.plot(range(0, horizon), pp, label="physics+ts")
    ax.set_title("Forecast comparison (one sample)")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "status": "ok",
        "metrics": {"mse_supervised": mse_sup, "mse_physics_fused": mse_phys},
        "artifacts": {"forecast_plot_png_b64": img_b64},
        "notes": {"equation": "x'' + 2 zeta w0 x' + w0^2 x = 0 (discrete residual penalty)"},
    }
