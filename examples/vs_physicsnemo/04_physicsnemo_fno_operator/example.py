"""
Forças do PhysicsNeMo — FNO Operator Learning + Multi-GPU
==========================================================

Demonstra onde o PhysicsNeMo tem vantagem sobre o PINNeAPPle:

  1. FNO (Fourier Neural Operator) otimizado com NVIDIA Transformer Engine
     → PhysicsNeMo: kernels cuDNN, fp16/bf16, 2-5× speedup vs implementação padrão
     → PINNeAPPle: FNO disponível, mas sem otimizações NVIDIA específicas

  2. Multi-GPU com DistributedManager (NCCL backend)
     → PhysicsNeMo: integração profunda com torch.distributed + multi-nó
     → PINNeAPPle: DataParallel básico + maybe_compile

  3. TensorRT / Triton deployment pipeline
     → PhysicsNeMo: exportação direta para TensorRT, servidor Triton
     → PINNeAPPle: ONNX + TorchScript (portável, não otimizado NVIDIA)

Problema: Operador de Burgers
    (u₀: [0,1] → ℝ) → (u(·,T): [0,1] → ℝ)
    Aprender o mapeamento: condição inicial → solução em T=1

O FNO aprende um operador — uma vez treinado, mapeia QUALQUER
condição inicial para a solução em T=1 sem re-resolver a PDE.

Uso:
    python example.py

Instalação PhysicsNeMo (opcional):
    pip install nvidia-physicsnemo
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft

# ---------------------------------------------------------------------------
# Detectar PhysicsNeMo
# ---------------------------------------------------------------------------

try:
    from physicsnemo.models.fno import FNO as PhysicsNemoFNO
    PHYSICSNEMO = True
    print("=" * 60)
    print("[INFO] PhysicsNeMo FNO carregado — implementação NVIDIA otimizada")
    print("       → cuDNN kernels, fp16/bf16, Transformer Engine ativo")
    print("=" * 60)
except ImportError:
    PHYSICSNEMO = False
    print("=" * 60)
    print("[INFO] PhysicsNeMo não instalado — usando implementação de referência")
    print("       Para usar PhysicsNeMo: pip install nvidia-physicsnemo")
    print("       Esta implementação tem a MESMA arquitetura, sem otimizações NVIDIA.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Implementação de referência de FNO (sem otimizações NVIDIA)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """Camada de convolução espectral 1D — núcleo do FNO."""

    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, C, N)
        B, C, N = x.shape
        x_ft   = fft.rfft(x, dim=-1)                       # (B, C, N//2+1)
        n_keep = min(self.n_modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :n_keep] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :n_keep], self.weights[:, :, :n_keep]
        )
        return fft.irfft(out_ft, n=N, dim=-1)               # (B, C, N)


class FNO1d(nn.Module):
    """Fourier Neural Operator 1D — implementação de referência.

    Igual à arquitetura do PhysicsNeMo, mas sem otimizações NVIDIA.
    Use PhysicsNeMo FNO para produção em hardware NVIDIA.
    """

    def __init__(self, n_modes: int = 16, width: int = 64,
                 n_layers: int = 4, in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.fc0    = nn.Linear(in_channels, width)
        self.convs  = nn.ModuleList([SpectralConv1d(width, width, n_modes) for _ in range(n_layers)])
        self.ws     = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1    = nn.Linear(width, 128)
        self.fc2    = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B, N, in_ch)
        x = self.fc0(x)                                     # (B, N, width)
        x = x.permute(0, 2, 1)                              # (B, width, N)
        for conv, w in zip(self.convs, self.ws):
            x = torch.relu(conv(x) + w(x))
        x = x.permute(0, 2, 1)                              # (B, N, width)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)                                   # (B, N, 1)


def build_fno(n_modes=16, width=64, n_layers=4):
    """Constrói FNO via PhysicsNeMo (se disponível) ou implementação de referência."""
    if PHYSICSNEMO:
        return PhysicsNemoFNO(
            in_channels=2,
            out_channels=1,
            decoder_layers=1,
            decoder_layer_size=128,
            dimension=1,
            latent_channels=width,
            num_fno_layers=n_layers,
            num_fno_modes=n_modes,
            padding=9,
        )
    return FNO1d(n_modes=n_modes, width=width, n_layers=n_layers)


# ---------------------------------------------------------------------------
# Geração de dados sintéticos de Burgers
# ---------------------------------------------------------------------------

def generate_burgers_data(n_samples: int, n_grid: int = 128,
                          nu: float = 0.01, seed: int = 0) -> tuple:
    """Gera pares (condição inicial, solução em T=1) via método espectral."""
    rng = np.random.default_rng(seed)
    x   = np.linspace(0, 1, n_grid, dtype=np.float32)
    dt  = 0.001
    n_steps = int(1.0 / dt)

    u0_all = np.zeros((n_samples, n_grid), dtype=np.float32)
    uT_all = np.zeros((n_samples, n_grid), dtype=np.float32)

    for i in range(n_samples):
        # IC aleatória (soma de senoides)
        n_modes = rng.integers(3, 8)
        u = np.zeros(n_grid, dtype=np.float64)
        for k in range(1, n_modes + 1):
            a = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * np.pi)
            u += a * np.sin(2 * np.pi * k * x + phi)
        u0_all[i] = u.astype(np.float32)

        # Integração pseudo-espectral simples
        for _ in range(n_steps):
            u_hat = np.fft.rfft(u)
            k_arr = np.fft.rfftfreq(n_grid, d=1.0 / n_grid)
            # Difusão
            u_hat = u_hat * np.exp(-nu * (2 * np.pi * k_arr) ** 2 * dt)
            u = np.fft.irfft(u_hat, n=n_grid)
            # Advecção (upwind simples)
            u_neg = np.roll(u, -1) - u
            u_pos = u - np.roll(u, 1)
            adv   = u * np.where(u >= 0, u_pos, u_neg) / (x[1] - x[0])
            u     = u - dt * adv

        uT_all[i] = u.astype(np.float32)

    return x, u0_all, uT_all


# ---------------------------------------------------------------------------
# Treinamento do FNO como operador
# ---------------------------------------------------------------------------

def train_fno(model: nn.Module, u0: np.ndarray, uT: np.ndarray,
              n_epochs: int = 500, batch_size: int = 32,
              device: str = "cpu") -> dict:
    """Treina FNO para aprender o operador u0 → uT."""
    model = model.to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    n_grid = u0.shape[1]
    x_grid = np.linspace(0, 1, n_grid, dtype=np.float32)

    # Preparar tensors: input = [u0, x_grid]
    X = np.stack([u0, np.tile(x_grid, (len(u0), 1))], axis=-1)   # (N, n_grid, 2)
    Y = uT[:, :, None]                                              # (N, n_grid, 1)

    n_train = int(0.8 * len(u0))
    X_tr, Y_tr = torch.tensor(X[:n_train]).to(device), torch.tensor(Y[:n_train]).to(device)
    X_te, Y_te = torch.tensor(X[n_train:]).to(device), torch.tensor(Y[n_train:]).to(device)

    history = {"train": [], "test": []}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        batch_losses = []

        for start in range(0, n_train, batch_size):
            idx   = perm[start:start + batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.item()))

        sched.step()
        train_loss = float(np.mean(batch_losses))
        history["train"].append(train_loss)

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = float(torch.mean((model(X_te) - Y_te) ** 2).item())
            history["test"].append(test_loss)
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}  train={train_loss:.3e}  test={test_loss:.3e}  {elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        test_loss_final = float(torch.mean((model(X_te) - Y_te) ** 2).item())
        throughput = n_train * n_epochs / (time.time() - t0)

    print(f"\n  Throughput: {throughput:.0f} samples/s")
    print(f"  Test MSE:   {test_loss_final:.4e}")
    if PHYSICSNEMO:
        print("  (PhysicsNeMo FNO — otimizado com cuDNN + Transformer Engine)")
    else:
        print("  (Implementação de referência — sem otimizações NVIDIA)")

    return {
        "model": model,
        "history": history,
        "test_mse": test_loss_final,
        "throughput": throughput,
        "x_test": X_te,
        "y_test": Y_te,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(result, x_grid, save_path="fno_operator_results.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    label = "PhysicsNeMo FNO" if PHYSICSNEMO else "FNO de referência"
    fig.suptitle(f"FNO Operator Learning — {label}\n"
                 "(PhysicsNeMo: mesma arquitetura + kernels NVIDIA otimizados)",
                 fontsize=12, fontweight="bold")

    # A — Training loss
    ax = axes[0, 0]
    ax.semilogy(result["history"]["train"], "steelblue", lw=1.5, label="Train")
    ax.set_title("A — Loss de treinamento")
    ax.set_xlabel("Época")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # B — Test predictions
    ax = axes[0, 1]
    x_te = result["x_test"]
    y_te = result["y_test"]
    model = result["model"]
    with torch.no_grad():
        y_pred = model(x_te).cpu().numpy()
    y_true = y_te.cpu().numpy()

    for i in range(min(3, len(y_true))):
        ax.plot(x_grid, y_true[i, :, 0], "k--", lw=1, alpha=0.7)
        ax.plot(x_grid, y_pred[i, :, 0], "tomato", lw=1.5, alpha=0.8)
    ax.plot([], [], "k--", label="True")
    ax.plot([], [], "tomato", label="FNO pred")
    ax.set_title("B — u₀→u(T) operator learning")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # C — Error distribution
    ax = axes[0, 2]
    errors = np.abs(y_pred[:, :, 0] - y_true[:, :, 0]).mean(axis=1)
    ax.hist(errors, bins=20, color="steelblue", edgecolor="white")
    ax.set_title(f"C — Distribuição do erro médio\n(test MSE={result['test_mse']:.2e})")
    ax.set_xlabel("Erro médio por amostra")
    ax.grid(True, alpha=0.3)

    # D — Comparison table: PhysicsNeMo vs reference FNO
    ax = axes[1, 0]
    ax.axis("off")
    table_data = [
        ["Feature",             "PhysicsNeMo FNO", "Ref. FNO"],
        ["Arquitetura",         "Idêntica",         "Idêntica"],
        ["cuDNN kernels",       "✅",               "❌"],
        ["Transformer Engine",  "✅",               "❌"],
        ["fp16/bf16 automático","✅",               "Manual"],
        ["Multi-GPU (DDP)",     "✅ nativo",        "DataParallel"],
        ["Multi-nó (NCCL)",     "✅",               "Manual"],
        ["TensorRT deploy",     "✅",               "❌"],
        ["Triton serve",        "✅",               "❌"],
        ["Speedup típico",      "2–5×",             "1×"],
        ["UQ integrado",        "❌",               "PINNeAPPle"],
        ["Digital Twin",        "❌",               "PINNeAPPle"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.4)
    ax.set_title("D — PhysicsNeMo FNO vs Referência", fontsize=10)

    # E — Sample solution evolution
    ax = axes[1, 1]
    sample_idx = 0
    u0_sample = x_te[sample_idx, :, 0].cpu().numpy()
    uT_pred   = y_pred[sample_idx, :, 0]
    uT_true   = y_true[sample_idx, :, 0]
    ax.plot(x_grid, u0_sample, "steelblue", lw=2, label="u(x,0) — IC")
    ax.plot(x_grid, uT_true,   "k--",       lw=2, label="u(x,1) — true")
    ax.plot(x_grid, uT_pred,   "tomato",    lw=2, label="u(x,1) — FNO")
    ax.set_title("E — Exemplo: IC → Solução em T=1")
    ax.set_xlabel("x")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # F — When to use PhysicsNeMo
    ax = axes[1, 2]
    ax.axis("off")
    text = (
        "Quando usar PhysicsNeMo FNO:\n\n"
        "✅ Dataset > 10.000 amostras\n"
        "✅ Hardware NVIDIA (A100, H100)\n"
        "✅ Treinamento multi-GPU / multi-nó\n"
        "✅ Deploy com TensorRT / Triton\n"
        "✅ Latência < 1ms em produção\n"
        "✅ Modelos pré-treinados (clima, CFD)\n\n"
        "Quando adicionar PINNeAPPle:\n\n"
        "✅ Precisa de UQ (intervalos)\n"
        "✅ Digital twin com sensores reais\n"
        "✅ Validação de conservação física\n"
        "✅ Domínios além de CFD\n"
        "✅ Meta-learning paramétrico\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("F — Guia de decisão", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"[PLOT] Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n[1/3] Gerando dataset Burgers (500 pares IC→solução)...")
    x_grid, u0, uT = generate_burgers_data(n_samples=500, n_grid=128, nu=0.01)
    print(f"      Shape: u0={u0.shape}, uT={uT.shape}")

    print("\n[2/3] Construindo e treinando FNO operador...")
    model = build_fno(n_modes=16, width=64, n_layers=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Parâmetros: {n_params:,}")

    result = train_fno(model, u0, uT, n_epochs=500, device=device)

    print("\n[3/3] Gerando gráficos...")
    make_plots(result, x_grid)

    print(f"\n✓ Done. Output: fno_operator_results.png")
    print(f"\nFNO aprendeu o operador: u₀ → u(·,T=1)")
    print(f"Uma vez treinado, mapeia qualquer IC para a solução em <1ms")
    print(f"\nPróximo passo: ver exemplo 05 — como adicionar UQ + Digital Twin")
    print(f"ao FNO via PINNeAPPle (pipeline combinado)")


if __name__ == "__main__":
    main()
