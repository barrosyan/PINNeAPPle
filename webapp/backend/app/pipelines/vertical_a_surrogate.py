from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def _heat2d_fdm(nx: int, ny: int, nt: int, dt: float, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple heat2d on [0,1]^2, Dirichlet boundary u=0.
    Returns:
      xs: (N,2)
      ts: (nt,)
      u:  (nt, ny, nx)
      mask: (ny,nx) all True
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    u = np.zeros((nt, ny, nx), dtype=np.float32)
    # IC: gaussian
    u0 = np.exp(-80.0 * ((X - 0.35) ** 2 + (Y - 0.65) ** 2)).astype(np.float32)
    u[0] = u0
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    cx = alpha * dt / (dx * dx)
    cy = alpha * dt / (dy * dy)
    for k in range(0, nt - 1):
        uk = u[k]
        un = uk.copy()
        un[1:-1, 1:-1] = (
            uk[1:-1, 1:-1]
            + cx * (uk[1:-1, 2:] - 2 * uk[1:-1, 1:-1] + uk[1:-1, :-2])
            + cy * (uk[2:, 1:-1] - 2 * uk[1:-1, 1:-1] + uk[:-2, 1:-1])
        )
        # boundary 0
        un[0, :] = 0
        un[-1, :] = 0
        un[:, 0] = 0
        un[:, -1] = 0
        u[k + 1] = un
    xs = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1).astype(np.float32)
    ts = (np.arange(nt) * dt).astype(np.float32)
    mask = np.ones((ny, nx), dtype=bool)
    return xs, ts, u, mask


def run_vertical_a(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical A (Surrogate Engineering Platform)
    - Uses a real solver (FDM heat2d) to generate data
    - Trains a small surrogate u(x,y,t)
    """
    nx = int(cfg.get("nx", 64))
    ny = int(cfg.get("ny", 64))
    nt = int(cfg.get("nt", 50))
    dt = float(cfg.get("dt", 0.01))
    alpha = float(cfg.get("alpha", 0.01))
    n_train = int(cfg.get("n_train", 20000))
    n_test = int(cfg.get("n_test", 5000))
    epochs = int(cfg.get("epochs", 10))
    lr = float(cfg.get("lr", 1e-3))
    hidden = int(cfg.get("hidden", 128))
    depth = int(cfg.get("depth", 4))

    xs, ts, u, _ = _heat2d_fdm(nx, ny, nt, dt, alpha)
    # build dataset by sampling points across time
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    total_pts = xs.shape[0] * nt
    idx = rng.choice(total_pts, size=n_train + n_test, replace=False)
    t_idx = idx // xs.shape[0]
    x_idx = idx % xs.shape[0]
    X = np.concatenate([xs[x_idx], ts[t_idx, None]], axis=1)  # (N,3)
    Y = u[t_idx, :, :].reshape(nt, -1)[t_idx, x_idx][:, None]  # (N,1)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(Y[:n_train], dtype=torch.float32)
    X_test = torch.tensor(X[n_train:], dtype=torch.float32)
    y_test = torch.tensor(Y[n_train:], dtype=torch.float32)

    # simple MLP surrogate
    layers = []
    in_dim = 3
    for i in range(depth):
        layers.append(torch.nn.Linear(in_dim if i == 0 else hidden, hidden))
        layers.append(torch.nn.GELU())
    layers.append(torch.nn.Linear(hidden, 1))
    model = torch.nn.Sequential(*layers)

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda", True)) else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch_iter(Xt, yt, bs=2048):
        n = Xt.shape[0]
        perm = torch.randperm(n, device=Xt.device)
        for i in range(0, n, bs):
            j = perm[i:i+bs]
            yield Xt[j], yt[j]

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model.train()
    for ep in range(epochs):
        for xb, yb in batch_iter(X_train, y_train):
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = torch.mean((pred - y_test) ** 2).item()
        rel_l2 = (torch.linalg.norm(pred - y_test) / (torch.linalg.norm(y_test) + 1e-12)).item()

    # plot: compare a time slice
    t_plot = int(cfg.get("t_plot_idx", nt // 2))
    Xg = torch.tensor(np.concatenate([xs, np.full((xs.shape[0], 1), ts[t_plot], dtype=np.float32)], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        ug = model(Xg).detach().cpu().numpy().reshape(ny, nx)
    ref = u[t_plot]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(ug - ref, origin="lower")
    ax.set_title("Error map (surrogate - solver) at t slice")
    fig.colorbar(im, ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "status": "ok",
        "metrics": {"mse": mse, "rel_l2": rel_l2},
        "artifacts": {"error_map_png_b64": img_b64},
        "notes": {
            "solver": "fdm_heat2d",
            "training_points": n_train,
            "test_points": n_test,
        },
    }
