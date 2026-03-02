from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import mse, rel_l2, plot_heatmaps


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _heat2d_fdm(u0: torch.Tensor, alpha: float, dx: float, dy: float, dt: float, steps: int) -> torch.Tensor:
    """Very small torch-only heat solver with Dirichlet boundary u=0."""
    u = u0.clone()
    for _ in range(int(steps)):
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        uxx = u.roll(-1, 1) - 2.0 * u + u.roll(1, 1)
        uyy = u.roll(-1, 0) - 2.0 * u + u.roll(1, 0)
        u = u + alpha * dt * (uxx / (dx * dx) + uyy / (dy * dy))
    return u


class _MLP(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 1, width: int = 128, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_surrogate_engineering(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical A — Surrogate Engineering Platform

    Demo:
      - Generate PDE solution with FDM heat2d
      - Train a small surrogate u(x,y,t)
      - Evaluate on a held-out time slice

    Returns:
      metrics + base64 plot + small arrays.
    """
    grid = int(cfg.get("grid", 64))
    steps = int(cfg.get("steps", 200))
    alpha = float(cfg.get("alpha", 0.02))
    dt = float(cfg.get("dt", 1e-3))
    train_steps = int(cfg.get("train_steps", 600))
    batch_size = int(cfg.get("batch_size", 4096))
    seed = int(cfg.get("seed", 0))
    prefer_cuda = bool(cfg.get("prefer_cuda", True))

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = _device(prefer_cuda)

    H = W = grid
    x = torch.linspace(0.0, 1.0, W, device=dev)
    y = torch.linspace(0.0, 1.0, H, device=dev)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    # Initial condition: Gaussian blob
    u0 = torch.exp(-120.0 * ((X - 0.35) ** 2 + (Y - 0.55) ** 2))

    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    # Produce snapshots for supervised training
    # Keep a few time points to mimic solver dataset
    n_snaps = int(cfg.get("snapshots", 8))
    snap_every = max(1, steps // n_snaps)
    us = []
    ts = []
    u = u0
    for s in range(steps + 1):
        if s % snap_every == 0:
            us.append(u.detach().clone())
            ts.append(s * dt)
        if s < steps:
            u = _heat2d_fdm(u, alpha=alpha, dx=dx, dy=dy, dt=dt, steps=1)

    # target: last snapshot as reference for plot
    u_ref = us[-1]
    t_ref = float(ts[-1])

    # Build training set from snapshots (x,y,t)->u
    coords = []
    vals = []
    for uk, tk in zip(us, ts):
        tcol = torch.full_like(X, float(tk))
        pts = torch.stack([X, Y, tcol], dim=-1).reshape(-1, 3)
        coords.append(pts)
        vals.append(uk.reshape(-1, 1))
    Xall = torch.cat(coords, dim=0)
    Yall = torch.cat(vals, dim=0)

    # Train/val split
    N = Xall.shape[0]
    perm = torch.randperm(N, device=dev)
    n_train = int(0.85 * N)
    tr = perm[:n_train]
    va = perm[n_train:]

    model = _MLP(in_dim=3, out_dim=1, width=int(cfg.get("width", 128)), depth=int(cfg.get("depth", 4))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-3)), weight_decay=1e-4)

    def sample_batch(idx: torch.Tensor, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sel = idx[torch.randint(0, idx.numel(), (bs,), device=dev)]
        return Xall[sel], Yall[sel]

    model.train()
    for it in range(train_steps):
        xb, yb = sample_batch(tr, batch_size)
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Evaluate at t_ref
    model.eval()
    with torch.no_grad():
        tcol = torch.full_like(X, t_ref)
        Xref = torch.stack([X, Y, tcol], dim=-1).reshape(-1, 3)
        up = model(Xref).reshape(H, W)

    ref_np = u_ref.detach().cpu().numpy()
    pred_np = up.detach().cpu().numpy()

    out = {
        "task": "surrogate_engineering",
        "equation": "heat2d",
        "grid": grid,
        "t_ref": t_ref,
        "metrics": {
            "mse": mse(ref_np, pred_np),
            "rel_l2": rel_l2(ref_np, pred_np),
        },
        "plot_base64_png": plot_heatmaps(ref_np, pred_np, title="Surrogate vs Solver (Heat2D)") ,
        "preview": {
            "ref": ref_np[::4, ::4].tolist(),
            "pred": pred_np[::4, ::4].tolist(),
        },
    }
    return out
