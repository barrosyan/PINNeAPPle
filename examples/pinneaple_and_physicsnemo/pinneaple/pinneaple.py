"""
PINNeAPPle Pure Benchmark — multi-architecture PINN comparison
==============================================================

Architectures benchmarked:
- VanillaPINN (pinneaple_models.pinns.vanilla.VanillaPINN)
- VPINN      (pinneaple_models.pinns.vpinn.VPINN)
- PINNsFormer (pinneaple_models.pinns.pinnsformer.PINNsFormer wrapped as PINN)
- PIELM (pinneaple_models.pinns.pielm.PIELM) via a gradient-enabled wrapper
  NOTE: upstream PIELM.forward_tensor is decorated with @torch.no_grad(), which
  prevents training by gradient descent. We intentionally avoid it.

Physics:
ODE residual: d2u/dx2 + u = 0
BCs: u(0)=0, u(pi)=0, u'(0)=1
Data: u(x) ~ sin(x) on random points

Outputs:
- per-architecture JSON report
- optional CSV summary
- PNG plots (loss curves, u(x) vs true, residuals, bar charts, Pareto scatter)
"""

from __future__ import annotations

import os
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Callable, List

import torch
import torch.nn as nn

# Matplotlib (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- PINNeAPPle ---
from pinneaple_pinn.factory.pinn_factory import PINNProblemSpec, PINNFactory, PINN
from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_models.pinns.vpinn import VPINN
from pinneaple_models.pinns.pinnsformer import PINNsFormer
from pinneaple_models.pinns.pielm import PIELM

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.set_float32_matmul_precision("high")

# -----------------------------
# Config / Report
# -----------------------------
@dataclass
class BenchConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    steps: int = 2000
    lr: float = 1e-3

    # sampling
    n_col: int = 2048         # collocation points per step
    n_data: int = 1024        # supervised points per step
    n_bc: int = 256           # points per boundary condition
    seed: int = 123

    # loss weights (PINNFactory bucket weights)
    w_pde: float = 1.0
    w_cond: float = 10.0
    w_data: float = 1.0

    # reporting
    out_dir: str = "benchmarks/_out"
    save_csv: bool = True


@dataclass
class BenchResult:
    name: str
    steps: int
    lr: float
    train_time_s: float
    final_total: float
    final_pde: float
    final_conditions: float
    final_data: float
    test_mse_u: float
    test_mse_residual: float


@dataclass
class TrainHistory:
    steps: List[int]
    total: List[float]
    pde: List[float]
    conditions: List[float]
    data: List[float]


class FactoryTensorAdapter(nn.Module):
    """
    Adapta modelos PINNeAPPle que retornam PINNOutput para retornarem Tensor (out.y),
    mantendo inverse_params para compatibilidade com PINNFactory.
    """
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        if hasattr(base, "inverse_params"):
            self.inverse_params = getattr(base, "inverse_params")
        else:
            self.inverse_params = nn.ParameterDict()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out = self.base(*args, **kwargs)

        if torch.is_tensor(out):
            return out

        for attr in ("y", "pred", "logits", "out"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if torch.is_tensor(v):
                    return v

        raise TypeError(
            f"Model output type {type(out)} is not Tensor and has no known tensor attribute."
        )

# -----------------------------
# Problem definition (ODE)
# -----------------------------
X0 = 0.0
X1 = math.pi

def u_true(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def make_spec(cfg: BenchConfig) -> PINNProblemSpec:
    # ODE: u_xx + u = 0
    # Conditions:
    #   u(0) = 0
    #   u(pi) = 0
    #   u_x(0) = 1
    return PINNProblemSpec(
        pde_residuals=[
            "Derivative(u(x), x, 2) + u(x)"
        ],
        conditions=[
            {"name": "bc_u_x0",   "equation": "u(x)",                       "weight": 1.0},
            {"name": "bc_u_xpi",  "equation": "u(x)",                       "weight": 1.0},
            {"name": "bc_du_x0",  "equation": "Derivative(u(x), x) - 1.0",  "weight": 1.0},
        ],
        independent_vars=["x"],
        dependent_vars=["u"],
        inverse_params=[],
        loss_weights={"pde": cfg.w_pde, "conditions": cfg.w_cond, "data": cfg.w_data},
        verbose=False,
    )

# -----------------------------
# Sampling
# -----------------------------
def sample_uniform(n: int, lo: float, hi: float, device, dtype) -> torch.Tensor:
    x = lo + (hi - lo) * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    return x

def make_batch(cfg: BenchConfig, device, dtype) -> Dict[str, Any]:
    # collocation: x in (0,pi)
    x_col = sample_uniform(cfg.n_col, X0, X1, device, dtype)

    # supervised data
    x_data = sample_uniform(cfg.n_data, X0, X1, device, dtype)
    y_data = u_true(x_data)

    # conditions
    x0 = torch.full((cfg.n_bc, 1), float(X0), device=device, dtype=dtype, requires_grad=True)
    xpi = torch.full((cfg.n_bc, 1), float(X1), device=device, dtype=dtype, requires_grad=True)
    x0_du = torch.full((cfg.n_bc, 1), float(X0), device=device, dtype=dtype, requires_grad=True)

    batch = {
        "collocation": (x_col,),
        "conditions": [
            (x0,),     # u(0)=0
            (xpi,),    # u(pi)=0
            (x0_du,),  # u'(0)=1
        ],
        "data": ((x_data,), y_data),
    }
    return batch

# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def test_mse_u(model: nn.Module, device, dtype, n: int = 2048) -> float:
    model.eval()
    x = torch.linspace(X0, X1, n, device=device, dtype=dtype).unsqueeze(1)
    y = u_true(x)
    yhat = model(x)
    if not torch.is_tensor(yhat):
        for attr in ("y", "pred", "logits", "out"):
            if hasattr(yhat, attr):
                yhat = getattr(yhat, attr)
                break
    return float(torch.mean((yhat - y) ** 2).item())

def test_mse_residual(model: nn.Module, device, dtype, n: int = 2048) -> float:
    model.eval()
    x = torch.linspace(X0, X1, n, device=device, dtype=dtype).unsqueeze(1)
    x = x.clone().detach().requires_grad_(True)

    yhat = model(x)
    if not torch.is_tensor(yhat):
        yhat = yhat.y

    (dy_dx,) = torch.autograd.grad(
        yhat, x,
        grad_outputs=torch.ones_like(yhat),
        create_graph=True, retain_graph=True
    )
    (d2y_dx2,) = torch.autograd.grad(
        dy_dx, x,
        grad_outputs=torch.ones_like(dy_dx),
        create_graph=True, retain_graph=True
    )

    r = d2y_dx2 + yhat
    return float(torch.mean(r ** 2).detach().item())

# -----------------------------
# Visual helpers
# -----------------------------
def predict_on_grid(model: nn.Module, device, dtype, n: int = 2048):
    model.eval()
    with torch.no_grad():
        x = torch.linspace(X0, X1, n, device=device, dtype=dtype).unsqueeze(1)
        y_true = u_true(x)
        y_hat = model(x)
        if not torch.is_tensor(y_hat):
            for attr in ("y", "pred", "logits", "out"):
                if hasattr(y_hat, attr):
                    y_hat = getattr(y_hat, attr)
                    break
        return x.detach().cpu().numpy(), y_true.detach().cpu().numpy(), y_hat.detach().cpu().numpy()

def residual_on_grid(model: nn.Module, device, dtype, n: int = 2048):
    model.eval()
    x = torch.linspace(X0, X1, n, device=device, dtype=dtype).unsqueeze(1)
    x = x.clone().detach().requires_grad_(True)

    y_hat = model(x)
    if not torch.is_tensor(y_hat):
        y_hat = y_hat.y

    (dy_dx,) = torch.autograd.grad(
        y_hat, x, grad_outputs=torch.ones_like(y_hat),
        create_graph=True, retain_graph=True
    )
    (d2y_dx2,) = torch.autograd.grad(
        dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
        create_graph=True, retain_graph=True
    )
    r = d2y_dx2 + y_hat
    return x.detach().cpu().numpy(), r.detach().cpu().numpy()

def save_visuals(
    cfg: BenchConfig,
    device,
    dtype,
    results: List[BenchResult],
    histories: Dict[str, TrainHistory],
    models: Dict[str, nn.Module],
):
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Loss curves
    def plot_curve(key: str, title: str, ylabel: str, filename: str, logy: bool = True):
        plt.figure()
        for name, h in histories.items():
            ys = getattr(h, key)
            plt.plot(h.steps, ys, label=name)
        if logy:
            plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, filename), dpi=200)
        plt.close()

    plot_curve("total", "Training loss (total)", "loss", "loss_total.png", logy=True)
    plot_curve("pde", "Training loss (pde)", "loss", "loss_pde.png", logy=True)
    plot_curve("conditions", "Training loss (conditions)", "loss", "loss_conditions.png", logy=True)
    plot_curve("data", "Training loss (data)", "loss", "loss_data.png", logy=True)

    # u(x): true vs preds
    plt.figure()
    any_model = next(iter(models.values()))
    x_np, y_true_np, _ = predict_on_grid(any_model, device, dtype, n=2048)
    plt.plot(x_np, y_true_np, label="u_true", linewidth=2)
    for name, m in models.items():
        x2, _, y_hat = predict_on_grid(m, device, dtype, n=2048)
        plt.plot(x2, y_hat, label=name)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("u(x): true vs model predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "u_true_vs_preds.png"), dpi=200)
    plt.close()

    # Residual magnitude |r(x)|
    plt.figure()
    for name, m in models.items():
        xr, r = residual_on_grid(m, device, dtype, n=2048)
        plt.plot(xr, np.abs(r), label=name)
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("|residual(x)|")
    plt.title("Residual magnitude along domain (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "residual_abs.png"), dpi=200)
    plt.close()

    # Bar charts + scatter
    names = [r.name for r in results]
    mse_u = [r.test_mse_u for r in results]
    mse_r = [r.test_mse_residual for r in results]
    tsec  = [r.train_time_s for r in results]

    def barplot(values, title, ylabel, filename, logy=False):
        plt.figure()
        plt.bar(names, values)
        if logy:
            plt.yscale("log")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, filename), dpi=200)
        plt.close()

    barplot(mse_u, "Test MSE on u(x)", "MSE", "bar_test_mse_u.png", logy=True)
    barplot(mse_r, "Test MSE on residual", "MSE", "bar_test_mse_residual.png", logy=True)
    barplot(tsec,  "Training time", "seconds", "bar_train_time.png", logy=False)

    plt.figure()
    plt.scatter(tsec, mse_u)
    for i, name in enumerate(names):
        plt.annotate(name, (tsec[i], mse_u[i]))
    plt.yscale("log")
    plt.xlabel("train_time_s")
    plt.ylabel("test_mse_u (log)")
    plt.title("Time vs accuracy (Pareto view)")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "scatter_time_vs_mse_u.png"), dpi=200)
    plt.close()

    print("[VIS] Saved plots into:", cfg.out_dir)

# -----------------------------
# PIELM wrapper (gradient-enabled)
# -----------------------------
class PIELMGradWrapper(nn.Module):
    """
    Wrap PIELM so it behaves like a standard torch.nn.Module:
      forward(x) -> y tensor
    and exposes inverse_params (empty) for PINNFactory compatibility.

    We intentionally DO NOT call pielm.forward_tensor(), because upstream has @torch.no_grad()
    which blocks gradients (see pinneaple_models/pinns/pielm.py).
    """
    def __init__(self, pielm: PIELM):
        super().__init__()
        self.pielm = pielm
        self.inverse_params = nn.ParameterDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pielm.hidden(x)        # grad-enabled
        y = h @ self.pielm.Beta         # grad-enabled
        return y

# -----------------------------
# Training loop (shared)
# -----------------------------
def train_one(
    name: str,
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]],
    cfg: BenchConfig,
    device,
    dtype,
) -> Tuple[BenchResult, TrainHistory]:
    model = model.to(device=device)

    if not hasattr(model, "inverse_params"):
        setattr(model, "inverse_params", nn.ParameterDict())

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    # warm batch (forces any lazy init paths)
    _ = make_batch(cfg, device, dtype)

    t0 = time.time()
    last_comps = {"total": float("nan"), "pde": float("nan"), "conditions": float("nan"), "data": float("nan")}

    hist = TrainHistory(steps=[], total=[], pde=[], conditions=[], data=[])

    for step in range(1, cfg.steps + 1):
        batch = make_batch(cfg, device, dtype)

        opt.zero_grad(set_to_none=True)
        total, comps = loss_fn(model, batch)
        total.backward()
        opt.step()

        last_comps = comps

        hist.steps.append(step)
        hist.total.append(float(comps.get("total", float("nan"))))
        hist.pde.append(float(comps.get("pde", float("nan"))))
        hist.conditions.append(float(comps.get("conditions", float("nan"))))
        hist.data.append(float(comps.get("data", float("nan"))))

        if step % max(1, cfg.steps // 10) == 0 or step == 1:
            print(
                f"[{name:10s}] step={step:5d}/{cfg.steps} "
                f"total={comps.get('total',0):.3e} pde={comps.get('pde',0):.3e} "
                f"cond={comps.get('conditions',0):.3e} data={comps.get('data',0):.3e}"
            )

    train_time = time.time() - t0

    mse_u = test_mse_u(model, device, dtype)
    mse_r = test_mse_residual(model, device, dtype)

    result = BenchResult(
        name=name,
        steps=cfg.steps,
        lr=cfg.lr,
        train_time_s=train_time,
        final_total=float(last_comps.get("total", float("nan"))),
        final_pde=float(last_comps.get("pde", float("nan"))),
        final_conditions=float(last_comps.get("conditions", float("nan"))),
        final_data=float(last_comps.get("data", float("nan"))),
        test_mse_u=mse_u,
        test_mse_residual=mse_r,
    )
    return result, hist

# -----------------------------
# Main benchmark
# -----------------------------
def main():
    cfg = BenchConfig()

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

    # PINNFactory loss
    spec = make_spec(cfg)
    factory = PINNFactory(spec)
    loss_fn = factory.generate_loss_function()

    results: List[BenchResult] = []
    histories: Dict[str, TrainHistory] = {}
    trained_models: Dict[str, nn.Module] = {}

    # 1) VanillaPINN
    vanilla = FactoryTensorAdapter(
        VanillaPINN(in_dim=1, out_dim=1, hidden=[64, 64, 64], activation="tanh")
    )
    r, h = train_one("VanillaPINN", vanilla, loss_fn, cfg, device, dtype)
    results.append(r); histories[r.name] = h; trained_models[r.name] = vanilla

    # 2) VPINN
    vpinn = FactoryTensorAdapter(
        VPINN(in_dim=1, out_dim=1, hidden=(64, 64, 64), activation="tanh")
    )
    r, h = train_one("VPINN", vpinn, loss_fn, cfg, device, dtype)
    results.append(r); histories[r.name] = h; trained_models[r.name] = vpinn

    # 3) PINNsFormer (Transformer) wrapped as PINN
    seq_len = 64
    pinnsformer_net = PINNsFormer(
        in_dim=1,
        out_dim=1,
        seq_len=seq_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_len=512,
        learnable_pos_emb=True,
    )
    pinnsformer = PINN(neural_network=pinnsformer_net, inverse_params_names=[], initial_guesses=None)

    # IMPORTANT: enforce batch sizes multiple of seq_len for PINNsFormer
    cfg_tf = BenchConfig(**asdict(cfg))
    cfg_tf.n_col = max(seq_len, (cfg_tf.n_col // seq_len) * seq_len)
    cfg_tf.n_data = max(seq_len, (cfg_tf.n_data // seq_len) * seq_len)
    cfg_tf.n_bc = max(seq_len, (cfg_tf.n_bc // seq_len) * seq_len)

    r, h = train_one("PINNsFormer", pinnsformer, loss_fn, cfg_tf, device, dtype)
    results.append(r); histories[r.name] = h; trained_models[r.name] = pinnsformer

    # 4) PIELM (ELM-style) with grad-enabled wrapper
    pielm_core = PIELM(in_dim=1, out_dim=1, hidden_dim=512, activation="tanh", freeze_random=True).to(device)
    pielm = PIELMGradWrapper(pielm_core)
    r, h = train_one("PIELM(wrap)", pielm, loss_fn, cfg, device, dtype)
    results.append(r); histories[r.name] = h; trained_models[r.name] = pielm

    # Save JSON
    out_json = os.path.join(cfg.out_dir, "pinneaple_pure_benchmark_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\nSaved:", out_json)

    # Save CSV
    if cfg.save_csv and len(results) > 0:
        out_csv = os.path.join(cfg.out_dir, "pinneaple_pure_benchmark_results.csv")
        header = list(asdict(results[0]).keys())
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in results:
                row = [str(getattr(r, k)) for k in header]
                f.write(",".join(row) + "\n")
        print("Saved:", out_csv)

    # Pretty print
    print("\n=== SUMMARY ===")
    for r in sorted(results, key=lambda x: x.test_mse_u):
        print(
            f"{r.name:12s} | mse_u={r.test_mse_u:.3e} | mse_res={r.test_mse_residual:.3e} | "
            f"time={r.train_time_s:.1f}s | final_total={r.final_total:.3e}"
        )

    # Visuals
    save_visuals(cfg, device, dtype, results, histories, trained_models)


if __name__ == "__main__":
    main()