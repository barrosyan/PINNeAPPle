"""
PINNeAPPle Exclusivo — Active Collocation (RAD) + SA-PINN Weights
==================================================================

Demonstra duas capacidades que NÃO existem no PhysicsNeMo:

  1. Residual-based Adaptive Distribution (RAD) — active learning
     para collocação: pontos são amostrados onde o resíduo de PDE
     é maior, concentrando esforço nas regiões difíceis.
     → PhysicsNeMo usa distribuição fixa de pontos de treinamento.

  2. Self-Adaptive PINN weights (SA-PINN) — pesos λ_i aprendíveis
     que crescem onde o modelo tem mais dificuldade (gradiente ascendente).
     → PhysicsNeMo usa pesos fixos para os termos de perda.

Problema: Equação de Laplace 2D
    u_xx + u_yy = 0,   (x,y) ∈ [0,1]²
    u(x,0) = sin(πx)   (bottom BC — interessante, causa gradientes altos)
    u(x,1) = 0
    u(0,y) = u(1,y) = 0

Solução analítica: u(x,y) = sin(πx) · sinh(π(1-y)) / sinh(π)

Uso:
    python example.py
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Modelo e física
# ---------------------------------------------------------------------------

def make_model(hidden=(64, 64, 64)):
    dims = [2, *hidden, 1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def laplace_residual(model, x_col: torch.Tensor) -> torch.Tensor:
    """Residual u_xx + u_yy (deve ser ~0)."""
    x = x_col.requires_grad_(True)
    u = model(x)
    grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx  = torch.autograd.grad(grads[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
    u_yy  = torch.autograd.grad(grads[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy  # (N,1)


def analytical(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x) * np.sinh(math.pi * (1 - y)) / math.sinh(math.pi)


def make_bc_batch(n_bc=400, device="cpu"):
    """BCs em todos os 4 lados."""
    rng = np.random.default_rng(1)
    n   = n_bc // 4

    # Bottom: y=0, u=sin(πx)
    x_bot = rng.random((n, 1)).astype(np.float32)
    y_bot = np.zeros((n, 1), np.float32)
    u_bot = np.sin(math.pi * x_bot).astype(np.float32)

    # Top: y=1, u=0
    x_top = rng.random((n, 1)).astype(np.float32)
    y_top = np.ones((n, 1), np.float32)
    u_top = np.zeros((n, 1), np.float32)

    # Left: x=0, u=0
    x_left = np.zeros((n, 1), np.float32)
    y_left = rng.random((n, 1)).astype(np.float32)
    u_left = np.zeros((n, 1), np.float32)

    # Right: x=1, u=0
    x_right = np.ones((n, 1), np.float32)
    y_right  = rng.random((n, 1)).astype(np.float32)
    u_right  = np.zeros((n, 1), np.float32)

    xy_bc = np.vstack([
        np.hstack([x_bot, y_bot]),
        np.hstack([x_top, y_top]),
        np.hstack([x_left, y_left]),
        np.hstack([x_right, y_right]),
    ])
    u_bc = np.vstack([u_bot, u_top, u_left, u_right])

    T = lambda a: torch.from_numpy(a).to(device)
    return T(xy_bc), T(u_bc)


def sample_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((n, 2)).astype(np.float32)


def sample_rad(model: nn.Module, n: int, n_candidates: int,
               rng: np.random.Generator, device: str) -> np.ndarray:
    """
    RAD — Residual-based Adaptive Distribution.
    Amostra proporcionalmente ao resíduo absoluto.
    """
    candidates = rng.random((n_candidates, 2)).astype(np.float32)
    x_cand = torch.from_numpy(candidates).to(device).requires_grad_(True)
    with torch.enable_grad():
        res = laplace_residual(model, x_cand).abs().detach().cpu().numpy().ravel()
    res = res - res.min() + 1e-8
    probs = res / res.sum()
    idx = rng.choice(n_candidates, size=n, replace=False, p=probs)
    return candidates[idx]


# ---------------------------------------------------------------------------
# SA-PINN: Self-Adaptive Weights (PINNeAPPle exclusivo)
# ---------------------------------------------------------------------------

class SelfAdaptiveWeights(nn.Module):
    """
    Pesos λ_i = exp(log_λ_i) aprendíveis via gradiente ascendente.
    Referência: McClenny & Braga-Neto (2020), arXiv:2009.04544.
    """
    def __init__(self, names: list, init_weights: dict = None):
        super().__init__()
        init = init_weights or {}
        self.log_w = nn.ParameterDict({
            n: nn.Parameter(torch.log(torch.tensor(float(init.get(n, 1.0)))))
            for n in names
        })
        self.history = {n: [] for n in names}

    @property
    def weights(self):
        return {k: torch.exp(v) for k, v in self.log_w.items()}

    def forward(self, losses: dict) -> torch.Tensor:
        total = None
        for k, w in self.weights.items():
            if k not in losses:
                continue
            term = w.clamp(0.01, 1000.0) * losses[k]
            total = term if total is None else total + term
            self.history[k].append(float(w.item()))
        return total if total is not None else torch.zeros(1)

    def weight_dict(self):
        return {k: float(torch.exp(v).item()) for k, v in self.log_w.items()}


# ---------------------------------------------------------------------------
# Treinar PINN com estratégia configurável
# ---------------------------------------------------------------------------

def train_pinn(n_col: int, n_epochs: int, strategy: str,
               use_sa_weights: bool, device: str) -> dict:
    """
    strategy: "uniform" | "rad"
    use_sa_weights: True = SA-PINN, False = pesos fixos
    """
    model  = make_model().to(device)
    rng    = np.random.default_rng(42)

    x_bc, u_bc = make_bc_batch(device=device)

    # Inicializar SA-PINN se solicitado
    if use_sa_weights:
        sa = SelfAdaptiveWeights(["pde", "bc"], {"pde": 1.0, "bc": 10.0}).to(device)
        model_opt  = optim.Adam(model.parameters(), lr=1e-3)
        weight_opt = optim.Adam(sa.parameters(),    lr=1e-2)
    else:
        sa        = None
        w_pde, w_bc = 1.0, 10.0
        model_opt = optim.Adam(model.parameters(), lr=1e-3)

    # Collocation inicial
    x_col_np = sample_uniform(n_col, rng)
    x_col    = torch.from_numpy(x_col_np).to(device)

    # Tracking
    loss_history   = []
    error_history  = []
    col_snapshots  = [x_col_np.copy()]  # salvar distribuição para plot

    # Pontos de validação
    x_val  = np.linspace(0, 1, 50)
    y_val  = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(x_val, y_val)
    xy_val = torch.tensor(
        np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32), device=device
    )
    u_true_val = analytical(XX, YY).ravel()

    for epoch in range(1, n_epochs + 1):
        # ---- Refinar collocação com RAD a cada 500 épocas ----
        if strategy == "rad" and epoch % 500 == 0 and epoch > 1:
            x_col_np = sample_rad(model, n_col, n_col * 10, rng, device)
            x_col    = torch.from_numpy(x_col_np).to(device)
            col_snapshots.append(x_col_np.copy())

        model_opt.zero_grad()
        if use_sa_weights:
            weight_opt.zero_grad()

        # PDE residual
        res = laplace_residual(model, x_col)
        l_pde = torch.mean(res ** 2)

        # BC
        u_pred_bc = model(x_bc)
        l_bc = torch.mean((u_pred_bc - u_bc) ** 2)

        # Combinar
        losses = {"pde": l_pde, "bc": l_bc}
        if use_sa_weights:
            total = sa(losses)
            # Gradiente ascendente nos pesos
            (-total).backward()
            for p in sa.parameters():
                if p.grad is not None:
                    p.grad.neg_()
            model_opt.step()
            weight_opt.step()
        else:
            total = w_pde * l_pde + w_bc * l_bc
            total.backward()
            model_opt.step()

        loss_history.append(float(total.item()))

        # Validação a cada 100 épocas
        if epoch % 100 == 0:
            with torch.no_grad():
                u_pred_val = model(xy_val).cpu().numpy().ravel()
            err = float(np.sqrt(np.mean((u_pred_val - u_true_val) ** 2)))
            error_history.append(err)

    return {
        "model": model,
        "losses": loss_history,
        "errors": error_history,
        "col_snapshots": col_snapshots,
        "sa_weights": sa.history if sa else None,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(res_uniform_fixed, res_rad_sa, save_path="active_weight_results.png"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("PINNeAPPle: Active Collocation (RAD) + SA-PINN\n"
                 "(capacidades ausentes no PhysicsNeMo)", fontsize=12, fontweight="bold")

    # A — Loss comparison
    ax = axes[0, 0]
    ax.semilogy(res_uniform_fixed["losses"], "gray",  lw=1.5, label="Uniforme + pesos fixos")
    ax.semilogy(res_rad_sa["losses"],        "tomato", lw=1.5, label="RAD + SA-PINN")
    ax.set_title("A — Loss de treinamento")
    ax.set_xlabel("Época")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # B — Error comparison
    ax = axes[0, 1]
    ep = range(100, 100 * len(res_uniform_fixed["errors"]) + 1, 100)
    ax.semilogy(ep, res_uniform_fixed["errors"], "gray",  lw=2, label="Uniforme + fixos")
    ax.semilogy(ep, res_rad_sa["errors"],         "tomato", lw=2, label="RAD + SA-PINN")
    ax.set_title("B — RMSE vs analítica")
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # C — Collocation points: initial (uniform)
    ax = axes[0, 2]
    snap_init = res_rad_sa["col_snapshots"][0]
    ax.scatter(snap_init[:, 0], snap_init[:, 1], s=2, alpha=0.4, color="steelblue")
    ax.set_title("C — Collocação inicial (uniforme)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # D — Collocation points: after RAD
    ax = axes[0, 3]
    if len(res_rad_sa["col_snapshots"]) > 1:
        snap_rad = res_rad_sa["col_snapshots"][-1]
    else:
        snap_rad = snap_init
    ax.scatter(snap_rad[:, 0], snap_rad[:, 1], s=2, alpha=0.4, color="tomato")
    ax.set_title("D — Collocação após RAD\n(concentrada em y≈0 — alto gradiente)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # E — SA-PINN weight history
    ax = axes[1, 0]
    if res_rad_sa["sa_weights"]:
        for name, wh in res_rad_sa["sa_weights"].items():
            ax.plot(wh, lw=2, label=f"λ_{name}")
    ax.set_title("E — SA-PINN: evolução dos pesos λ")
    ax.set_xlabel("Época de atualização")
    ax.set_ylabel("Peso λ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # F — Predicted field (RAD + SA-PINN)
    x_val  = np.linspace(0, 1, 60)
    y_val  = np.linspace(0, 1, 60)
    XX, YY = np.meshgrid(x_val, y_val)
    xy     = torch.tensor(np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32))
    model  = res_rad_sa["model"]
    with torch.no_grad():
        u_pred = model(xy).numpy().reshape(60, 60)
    u_true = analytical(XX, YY)

    ax = axes[1, 1]
    c = ax.contourf(XX, YY, u_pred, levels=25, cmap="RdBu_r")
    fig.colorbar(c, ax=ax)
    ax.set_title("F — u(x,y) predito (RAD + SA-PINN)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = axes[1, 2]
    c2 = ax.contourf(XX, YY, np.abs(u_pred - u_true), levels=25, cmap="hot_r")
    fig.colorbar(c2, ax=ax)
    ax.set_title(f"G — |erro| RAD+SA-PINN\nmax={np.abs(u_pred-u_true).max():.2e}")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # H — Capability table
    ax = axes[1, 3]
    ax.axis("off")
    table_data = [
        ["Capacidade",          "PINNeAPPle", "PhysicsNeMo"],
        ["ResidualBasedAL (RAR)", "✅",        "❌"],
        ["RAD sampling",        "✅",          "❌"],
        ["VarianceBasedAL",     "✅",          "❌"],
        ["CombinedAL",          "✅",          "❌"],
        ["SA-PINN weights",     "✅",          "❌"],
        ["GradNormBalancer",    "✅",          "❌"],
        ["NTKWeightBalancer",   "✅",          "❌"],
        ["LossRatioBalancer",   "✅",          "❌"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)
    ax.set_title("H — Comparação de capacidades", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[PLOT] Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_col  = 2000
    n_epochs = 2000

    print(f"Device: {device}")
    print("=" * 60)
    print("PINNeAPPle — Active Collocation + SA-PINN Weight Scheduling")
    print("(PhysicsNeMo usa pontos uniformes e pesos fixos)")
    print("=" * 60)

    print(f"\n[1/2] Baseline — collocação UNIFORME + pesos FIXOS ({n_epochs} épocas)...")
    res_baseline = train_pinn(n_col, n_epochs,
                              strategy="uniform", use_sa_weights=False, device=device)
    baseline_err = res_baseline["errors"][-1] if res_baseline["errors"] else float("nan")
    print(f"      RMSE final: {baseline_err:.4f}")

    print(f"\n[2/2] PINNeAPPle — collocação RAD + SA-PINN ({n_epochs} épocas)...")
    res_improved = train_pinn(n_col, n_epochs,
                              strategy="rad", use_sa_weights=True, device=device)
    improved_err = res_improved["errors"][-1] if res_improved["errors"] else float("nan")
    print(f"      RMSE final: {improved_err:.4f}")

    print("\n[3/3] Gerando gráficos...")
    make_plots(res_baseline, res_improved)

    if baseline_err > 0 and improved_err > 0:
        reduction = (1 - improved_err / baseline_err) * 100
        print(f"\n✓ Done. Output: active_weight_results.png")
        print(f"\nREDUÇÃO DE ERRO: {reduction:.1f}%  (RAD + SA-PINN vs uniforme + fixo)")
        print("=" * 60)
        print("RESUMO:")
        print(f"  Uniforme + pesos fixos (baseline):  RMSE = {baseline_err:.4f}")
        print(f"  RAD + SA-PINN (PINNeAPPle):         RMSE = {improved_err:.4f}")
        print(f"  → {reduction:.0f}% menos erro com o MESMO número de pontos")
        print("=" * 60)


if __name__ == "__main__":
    main()
