from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import mse, rel_l2, plot_heatmaps


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _heat2d_solution(X: torch.Tensor, Y: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    # Manufactured solution u = exp(-2*pi^2*alpha*t) * sin(pi x) sin(pi y)
    return torch.exp(-2.0 * (torch.pi**2) * alpha * t) * torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)


def _pde_residual_heat(u: torch.Tensor, xyt: torch.Tensor, alpha: float) -> torch.Tensor:
    # u_t - alpha*(u_xx + u_yy)
    xyt = xyt.requires_grad_(True)
    u = u
    grads = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_t - alpha * (u_xx + u_yy)


class Linear(nn.Module):
    def __init__(self, in_dim=3, out_dim=1):
        super().__init__()
        self.l = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.l(x)


class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=128, depth=4, act="gelu"):
        super().__init__()
        A = nn.GELU if act == "gelu" else nn.Tanh
        layers = [nn.Linear(in_dim, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=128, blocks=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(blocks)])
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.inp(x))
        for b in self.blocks:
            h = h + 0.5 * b(h)
        return self.out(h)


class FourierMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=128, depth=3, n_freq=16, sigma=10.0):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        self.register_buffer("B", B)
        ff_dim = 2 * n_freq
        self.mlp = MLP(in_dim=ff_dim, out_dim=out_dim, width=width, depth=depth, act="gelu")

    def forward(self, x):
        # x: (N,3)
        proj = 2.0 * torch.pi * (x @ self.B)
        z = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.mlp(z)


class SIREN(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=128, depth=4, w0=30.0):
        super().__init__()
        self.w0 = float(w0)
        self.inp = nn.Linear(in_dim, width)
        self.hid = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, out_dim)
        self._init_siren()

    def _init_siren(self):
        with torch.no_grad():
            self.inp.weight.uniform_(-1 / self.inp.in_features, 1 / self.inp.in_features)
            for l in self.hid:
                l.weight.uniform_(-np.sqrt(6 / l.in_features) / self.w0, np.sqrt(6 / l.in_features) / self.w0)

    def forward(self, x):
        h = torch.sin(self.w0 * self.inp(x))
        for l in self.hid:
            h = torch.sin(self.w0 * l(h))
        return self.out(h)


@dataclass
class ModelEntry:
    name: str
    model: nn.Module
    supports_physics: bool


def _make_models(dev: torch.device) -> List[ModelEntry]:
    return [
        ModelEntry("linear", Linear().to(dev), False),
        ModelEntry("mlp", MLP(width=128, depth=4).to(dev), False),
        ModelEntry("res_mlp", ResMLP(width=128, blocks=4).to(dev), False),
        ModelEntry("fourier_mlp_pinn", FourierMLP(width=128, depth=3).to(dev), True),
        ModelEntry("siren_pinn", SIREN(width=128, depth=4).to(dev), True),
    ]


def run_benchmark_arena(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical C — Scientific Benchmark Arena

    Demo benchmark: Heat2D manufactured solution.
    - Data: supervised samples
    - PINN models additionally minimize PDE residual + BC/IC

    Outputs:
      - ranking table
      - best model plot
    """
    prefer_cuda = bool(cfg.get("prefer_cuda", True))
    dev = _device(prefer_cuda)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    alpha = float(cfg.get("alpha", 0.02))
    n_data = int(cfg.get("n_data", 20000))
    n_col = int(cfg.get("n_col", 8000))
    n_bc = int(cfg.get("n_bc", 4000))
    n_ic = int(cfg.get("n_ic", 4000))

    train_steps = int(cfg.get("train_steps", 1200))
    batch = int(cfg.get("batch_size", 2048))
    lr = float(cfg.get("lr", 2e-3))

    w_data = float(cfg.get("w_data", 1.0))
    w_pde = float(cfg.get("w_pde", 1.0))
    w_bc = float(cfg.get("w_bc", 5.0))
    w_ic = float(cfg.get("w_ic", 5.0))

    # Sample supervised data in domain [0,1]^2 x [0,1]
    x = torch.rand(n_data, 1, device=dev)
    y = torch.rand(n_data, 1, device=dev)
    t = torch.rand(n_data, 1, device=dev)
    xyt = torch.cat([x, y, t], dim=1)
    u = _heat2d_solution(x, y, t, alpha).reshape(-1, 1)

    # Collocation points
    xc = torch.rand(n_col, 1, device=dev)
    yc = torch.rand(n_col, 1, device=dev)
    tc = torch.rand(n_col, 1, device=dev)
    xyt_c = torch.cat([xc, yc, tc], dim=1)

    # Boundary points (x=0,1 or y=0,1)
    rb = torch.rand(n_bc, 1, device=dev)
    tb = torch.rand(n_bc, 1, device=dev)
    side = torch.randint(0, 4, (n_bc, 1), device=dev)
    xb = torch.where(side == 0, torch.zeros_like(rb), torch.where(side == 1, torch.ones_like(rb), rb))
    yb = torch.where(side == 2, torch.zeros_like(rb), torch.where(side == 3, torch.ones_like(rb), rb))
    xyt_b = torch.cat([xb, yb, tb], dim=1)
    ub = torch.zeros(n_bc, 1, device=dev)  # sin(pi x)sin(pi y)=0 on boundary

    # Initial condition at t=0
    xi = torch.rand(n_ic, 1, device=dev)
    yi = torch.rand(n_ic, 1, device=dev)
    ti = torch.zeros(n_ic, 1, device=dev)
    xyt_i = torch.cat([xi, yi, ti], dim=1)
    ui = _heat2d_solution(xi, yi, ti, alpha).reshape(-1, 1)

    # Test grid at t=0.6
    grid = int(cfg.get("grid", 64))
    tg = float(cfg.get("t_test", 0.6))
    xs = torch.linspace(0.0, 1.0, grid, device=dev)
    ys = torch.linspace(0.0, 1.0, grid, device=dev)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    Tt = torch.full_like(X, tg)
    xyt_test = torch.stack([X, Y, Tt], dim=-1).reshape(-1, 3)
    u_test = _heat2d_solution(X, Y, Tt, alpha).reshape(-1, 1)

    def rand_batch(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, X.shape[0], (batch,), device=dev)
        return X[idx], Y[idx]

    results = []
    models = _make_models(dev)

    for entry in models:
        m = entry.model
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-6)
        m.train()
        for it in range(train_steps):
            xd, yd = rand_batch(xyt, u)
            pred = m(xd)
            loss = w_data * torch.mean((pred - yd) ** 2)

            if entry.supports_physics:
                # PDE residual
                xc_b = xyt_c[torch.randint(0, xyt_c.shape[0], (batch,), device=dev)]
                uc = m(xc_b)
                r = _pde_residual_heat(uc, xc_b, alpha)
                loss = loss + w_pde * torch.mean(r ** 2)

                # BC
                xb_b, ub_b = rand_batch(xyt_b, ub)
                pb = m(xb_b)
                loss = loss + w_bc * torch.mean((pb - ub_b) ** 2)

                # IC
                xi_b, ui_b = rand_batch(xyt_i, ui)
                pi = m(xi_b)
                loss = loss + w_ic * torch.mean((pi - ui_b) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Evaluate
        m.eval()
        with torch.no_grad():
            up = m(xyt_test)
        ref = u_test.detach().cpu().numpy().reshape(grid, grid)
        pred = up.detach().cpu().numpy().reshape(grid, grid)
        results.append({
            "model": entry.name,
            "supports_physics_loss": entry.supports_physics,
            "mse": mse(ref, pred),
            "rel_l2": rel_l2(ref, pred),
        })

    # Rank by rel_l2 then mse
    results_sorted = sorted(results, key=lambda r: (r["rel_l2"], r["mse"]))
    best_name = results_sorted[0]["model"]

    # Recompute best model prediction for plot
    best_entry = next(e for e in models if e.name == best_name)
    best_entry.model.eval()
    with torch.no_grad():
        best_pred = best_entry.model(xyt_test).detach().cpu().numpy().reshape(grid, grid)
    best_ref = u_test.detach().cpu().numpy().reshape(grid, grid)

    return {
        "task": "scientific_benchmark_arena",
        "problem": "heat2d_manufactured",
        "ranking": results_sorted,
        "best_model": best_name,
        "plot_base64_png": plot_heatmaps(best_ref, best_pred, title=f"Arena Best: {best_name}") ,
    }
