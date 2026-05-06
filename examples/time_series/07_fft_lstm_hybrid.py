"""
07_fft_lstm_hybrid.py — Hybrid FFT Decomposition + LSTM Residual Forecasting

Pipeline overview
-----------------
0. Reproducibility: global seed fixed.
1. Generation and saving of a synthetic multivariate CSV dataset.
2. Loading the CSV via the PINNeAPPle `csv_loader` (works with ANY CSV).
3. FFT analysis: power spectrum and dominant periods per variable.
4. FFT decomposition fitted ONLY on the training set (no leakage).
5. Residual computation for the entire dataset.
6. Construction of TSDataModules for the hybrid model and for the pure LSTM.
7. Training with anti-overfitting strategies: dropout, weight_decay, grad_clip,
   EarlyStopping, and proper temporal split.
8. Rolling evaluation on the test set (holdout never seen during training).
9. Fair comparison between 4 approaches:
   • Naive (repeat last value)
   • FFT-only (deterministic extrapolation)
   • Pure LSTM (trained on normalized raw data)
   • Hybrid FFT + LSTM (trained on residuals)

## Realism philosophy

Models are NOT magical. Stochastic noise is irreducible: no honest model
will reach zero error. The expected metric reflects the Bayes floor imposed
by the additive noise in the synthetic data (~0.25–0.35 normalized MAE).
Any result below that would indicate overfitting or leakage.

Anti-leakage guarantees:

* TimeSeriesScaler fitted ONLY on the training set.
* FFTDecomposer fitted ONLY on the training set.
* EarlyStopping monitors val_loss (not test).
* The test set is never seen during training or validation.

External dependencies (besides PINNeAPPle):
pandas, numpy, torch, (optional) matplotlib for plots.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force UTF-8 output on Windows terminals (cp1252 can't encode many Unicode chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneaple_timeseries.csv_loader import generate_synthetic_csv, load_timeseries_csv
from pinneaple_timeseries.decomposition.fft_lstm import (
    FFTLSTMPipeline,
    ResidualLSTMConfig,
)
from pinneaple_timeseries.datamodule import TSDataModule
from pinneaple_timeseries.spec import TimeSeriesSpec
from pinneaple_timeseries.datasets.windowed import WindowedTimeSeriesDataset
from pinneaple_timeseries.metrics_ext.point import mae as np_mae, rmse as np_rmse, smape as np_smape

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import regression_metrics_bundle
from pinneaple_train.callbacks import EarlyStopping, ModelCheckpoint
from pinneaple_train.preprocess import (
    PreprocessPipeline,
    MissingValueStep,
    RobustScaleStep,
)


# ============================================================
# 0) Reprodutibilidade
# ============================================================
SEED = 42
INPUT_LEN = 96    # janela de contexto para o LSTM
HORIZON   = 24    # passos à frente para prever
BATCH     = 64
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"Dispositivo: {DEVICE}")


# ============================================================
# 1) Gerar e salvar CSV sintético
# ============================================================
CSV_PATH = Path("data") / "synthetic_multivariate.csv"

print("\n=== [1] Gerando dataset CSV sintético ===")
generate_synthetic_csv(CSV_PATH, T=1000, seed=SEED, freq="D")
print(f"CSV salvo em: {CSV_PATH.resolve()}")


# ============================================================
# 2) Carregar CSV via PINNeAPPle
# ============================================================
print("\n=== [2] Carregando CSV ===")
result = load_timeseries_csv(
    CSV_PATH,
    target_cols=["y1", "y2", "y3"],
    time_col="date",
    fill_method="linear",
    normalize=True,         # z-score ajustado APENAS no treino
    train_ratio=0.70,       # 700 steps
    val_ratio=0.15,         # 150 steps
)                           # test_ratio = 0.15 (150 steps) — nunca visto no treino

series_tensor = result["series_tensor"]   # (1000, 3) — série completa normalizada
train_tensor  = result["train_tensor"]    # (700, 3)
val_tensor    = result["val_tensor"]      # (150, 3)
test_tensor   = result["test_tensor"]     # (150, 3)
scaler        = result["scaler"]
meta          = result["meta"]

T, F = meta["T"], meta["F"]
n_train, n_val, n_test = meta["n_train"], meta["n_val"], meta["n_test"]

print(f"Série: T={T} | F={F} colunas: {meta['columns']}")
print(f"Splits: treino={n_train} | val={n_val} | teste={n_test} steps")
print(f"Normalizado: {meta['normalized']}")


# ============================================================
# 3) Análise FFT — espectro de potência e períodos dominantes
# ============================================================
print("\n=== [3] Análise FFT dos dados de treino ===")

fft_pipeline = FFTLSTMPipeline(
    n_harmonics=12,
    detrend=True,
    auto_harmonics=True,    # seleciona automaticamente harmônicos relevantes
    auto_threshold=0.05,
    input_len=INPUT_LEN,
    horizon=HORIZON,
    n_features=F,
    n_targets=F,
    hidden_size=128,
    num_layers=2,
    dropout=0.20,           # dropout anti-overfitting
)

# Espectro de potência dos dados de treino
spectra = fft_pipeline.power_spectrum(train_tensor.numpy())
for i, (col, sp) in enumerate(zip(meta["columns"], spectra)):
    top3_idx = np.argsort(sp["power"])[::-1][:3]
    top3_periods = sp["periods"][top3_idx]
    top3_power   = sp["power"][top3_idx]
    # Filtra infinitos (período DC)
    top3_periods = top3_periods[np.isfinite(top3_periods)][:3]
    periods_str  = ", ".join(f"{p:.1f}" for p in top3_periods)
    print(f"  Canal '{col}': períodos dominantes ≈ [{periods_str}] steps")


# ============================================================
# 4) Ajuste do FFTDecomposer SOMENTE nos dados de treino
#    (nenhuma informação de val/test usada aqui)
# ============================================================
print("\n=== [4] Ajustando FFTDecomposer no treino ===")
fft_pipeline.fit_fft(train_tensor.numpy())   # (700, 3) — SOMENTE treino

periods_per_channel = fft_pipeline.dominant_periods()
for col, periods in zip(meta["columns"], periods_per_channel):
    finite_periods = [p for p in periods if math.isfinite(p)][:4]
    print(f"  '{col}': {[f'{p:.1f}' for p in finite_periods]}")


# ============================================================
# 5) Calcular resíduos para TODA a série
#    (FFT reconstrói treino + extrapola val/test — sem leakage)
# ============================================================
print("\n=== [5] Calculando resíduos (original − FFT) ===")
residuals_tensor = fft_pipeline.residuals_tensor(series_tensor.numpy())  # (1000, 3)

res_train = residuals_tensor[:n_train]                    # (700, 3)
res_val   = residuals_tensor[n_train : n_train + n_val]   # (150, 3)
res_test  = residuals_tensor[n_train + n_val :]           # (150, 3)

print(f"  Resíduo treino — std por canal: {res_train.std(0).numpy().round(4)}")
print(f"  Resíduo val   — std por canal: {res_val.std(0).numpy().round(4)}")
print(f"  Série original treino — std : {train_tensor.std(0).numpy().round(4)}")
# O resíduo deve ter variância menor que a série original (FFT capturou tendência/sazonalidade)


# ============================================================
# 6) TSDataModules
#    — usa SOMENTE treino+val para criar janelas de treinamento
#    — teste permanece completamente fora dos DataLoaders
# ============================================================
spec = TimeSeriesSpec(input_len=INPUT_LEN, horizon=HORIZON, stride=1)

n_trainval = n_train + n_val  # 850 steps disponíveis para treino+val
val_frac   = n_val / n_trainval  # fração de val dentro de treino+val

# DataModule para o modelo HÍBRIDO (treinado nos resíduos)
residuals_trainval = residuals_tensor[:n_trainval]   # (850, 3)
dm_hybrid = TSDataModule(
    series=residuals_trainval,
    spec=spec,
    batch_size=BATCH,
    val_ratio=val_frac,
    num_workers=0,
)

# DataModule para o LSTM PURO (treinado na série normalizada bruta)
raw_trainval = series_tensor[:n_trainval]            # (850, 3)
dm_raw = TSDataModule(
    series=raw_trainval,
    spec=spec,
    batch_size=BATCH,
    val_ratio=val_frac,
    num_workers=0,
)

train_loader_h, val_loader_h = dm_hybrid.make_sequential_holdout_loaders()
train_loader_r, val_loader_r = dm_raw.make_sequential_holdout_loaders()

print(f"\n=== [6] DataModules criados ===")
print(f"  Janelas totais (treino+val): {len(dm_hybrid.dataset())}")
print(f"  Split val: {val_frac:.1%}")


# ============================================================
# 7) Definição dos modelos (plain nn.Module para compatibilidade
#    direta com o Trainer — retorna tensor, não TSOutput)
# ============================================================

class _LSTM(nn.Module):
    """LSTM com anti-overfitting embutido.

    Técnicas usadas:
      • LayerNorm na entrada (estabiliza gradientes, seguro com qualquer batch size)
      • Dropout inter-camadas no LSTM
      • LayerNorm no último hidden state
      • Dropout antes da cabeça linear
      • weight_decay e grad_clip configurados no TrainConfig
      • EarlyStopping monitorando val_loss
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, horizon: int):
        super().__init__()
        self.horizon    = horizon
        self.n_features = n_features

        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_norm  = nn.LayerNorm(hidden_size)
        self.head_drop  = nn.Dropout(p=dropout)
        self.head       = nn.Linear(hidden_size, horizon * n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)  →  retorna (B, H, F)
        xn = self.input_norm(x)
        out, _ = self.lstm(xn)
        last = self.post_norm(out[:, -1, :])
        last = self.head_drop(last)
        pred = self.head(last)
        return pred.view(x.shape[0], self.horizon, self.n_features)


def _build_lstm() -> _LSTM:
    return _LSTM(
        n_features=F,
        hidden_size=128,
        num_layers=2,
        dropout=0.20,      # dropout entre camadas e antes da cabeça
        horizon=HORIZON,
    )


# ============================================================
# 8) Configuração de treinamento (anti-overfitting)
# ============================================================

def _make_train_cfg(run_name: str) -> TrainConfig:
    return TrainConfig(
        epochs=150,            # EarlyStopping vai parar antes disso
        lr=3e-4,               # lr conservador (Adam)
        weight_decay=1e-4,     # L2 regularization
        grad_clip=1.0,         # gradient clipping (evita explosão de gradiente)
        amp=False,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        log_dir="runs",
        run_name=run_name,
        save_best=True,
    )

def _make_trainer(model: nn.Module, run_name: str) -> Trainer:
    Path("checkpoints").mkdir(exist_ok=True)
    return Trainer(
        model=model,
        loss_fn=CombinedLoss(supervised=SupervisedLoss("mse")),
        metrics=regression_metrics_bundle(),
        preprocess=PreprocessPipeline(steps=[
            MissingValueStep(key="x", strategy="ffill", enabled=True),
            RobustScaleStep(key="x", enabled=True),  # robustez a outliers residuais
        ]),
        early_stopping=EarlyStopping(
            patience=15,           # para se val_total não melhorar em 15 épocas
            monitor="val_total",   # chave usada internamente pelo Trainer
            mode="min",
        ),
        checkpoint=ModelCheckpoint(
            path=f"checkpoints/{run_name}.pt",
            monitor="val_total",
        ),
    )


# ============================================================
# 9) Treinar modelo HÍBRIDO (FFT resíduos → LSTM)
# ============================================================
print("\n=== [9] Treinando modelo HÍBRIDO FFT+LSTM ===")

hybrid_model = _build_lstm()
hybrid_trainer = _make_trainer(hybrid_model, "hybrid_fft_lstm")
hybrid_result  = hybrid_trainer.fit(train_loader_h, val_loader_h, _make_train_cfg("hybrid"))

print(f"  Melhor val_total: {hybrid_result.get('best_val', float('nan')):.6f}")
print(f"  Checkpoint: {hybrid_result.get('best_path', 'n/a')}")


# ============================================================
# 10) Treinar LSTM PURO (mesma arquitetura, dados brutos)
# ============================================================
print("\n=== [10] Treinando LSTM PURO (baseline neural) ===")

lstm_model   = _build_lstm()
lstm_trainer = _make_trainer(lstm_model, "pure_lstm")
lstm_result  = lstm_trainer.fit(train_loader_r, val_loader_r, _make_train_cfg("lstm"))

print(f"  Melhor val_total: {lstm_result.get('best_val', float('nan')):.6f}")
print(f"  Checkpoint: {lstm_result.get('best_path', 'n/a')}")


# ============================================================
# 11) Avaliação rolling no conjunto de TESTE
#
#  Protocolo correto e sem leakage:
#   • Para cada janela de teste (input_len=96, horizon=24):
#       – O INPUT pode usar steps anteriores ao início do teste
#         (contexto vindo do val), mas o TARGET é sempre futuro.
#       – O modelo NUNCA foi treinado nem validado nesses alvos.
#   • Métricas calculadas no espaço normalizado (z-score) para
#     comparação justa. Valores absolutos dependem da escala original.
# ============================================================

def rolling_evaluate_model(
    model: nn.Module,
    series: torch.Tensor,      # série completa (T, F)
    *,
    input_len: int,
    horizon:   int,
    test_start: int,           # índice do primeiro step do teste
    device:    str,
    mode: str = "raw",         # "raw" ou "residual" + "fft" para híbrido
    fft_pipeline_: FFTLSTMPipeline | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Avaliação rolling por janelas deslizantes no período de teste.

    Retorna (preds, trues) ambos com shape (N_windows, H, F).
    Cada window usa o input ANTERIOR ao target — sem ver o futuro.
    """
    model.eval()
    T_total = len(series)
    # Janelas onde o target está inteiramente dentro ou após test_start
    # e inteiramente dentro da série
    first_window = max(0, test_start - input_len)
    last_window  = T_total - input_len - horizon   # último índice válido

    if first_window > last_window:
        raise ValueError("test_start ou horizon incompatíveis com o tamanho da série.")

    preds, trues = [], []
    with torch.no_grad():
        for i in range(first_window, last_window + 1):
            x_raw = series[i : i + input_len]               # (L, F)
            y_true = series[i + input_len : i + input_len + horizon]  # (H, F)

            if mode == "residual":
                # Modelo treinado em resíduos; saída é o resíduo previsto
                x_input = residuals_tensor[i : i + input_len]       # (L, F)
                x_batch = x_input.unsqueeze(0).to(device)
                y_res   = model(x_batch).squeeze(0).cpu().numpy()    # (H, F)

                # Combinar com extrapolação FFT
                fft_part = fft_pipeline_.fft_extrapolation(
                    start=i + input_len,
                    horizon=horizon,
                )                                                     # (H, F)
                y_pred = fft_part + y_res                             # (H, F)

            else:  # mode == "raw"
                x_batch = x_raw.unsqueeze(0).to(device)
                y_pred  = model(x_batch).squeeze(0).cpu().numpy()    # (H, F)

            preds.append(y_pred)
            trues.append(y_true.numpy())

    return np.array(preds), np.array(trues)  # (N, H, F)


def rolling_evaluate_naive(
    series: torch.Tensor,
    *,
    input_len: int,
    horizon: int,
    test_start: int,
    kind: str = "naive",
) -> Tuple[np.ndarray, np.ndarray]:
    """Baseline Naive e Drift."""
    T_total = len(series)
    first_w = max(0, test_start - input_len)
    last_w  = T_total - input_len - horizon
    preds, trues = [], []

    for i in range(first_w, last_w + 1):
        x = series[i : i + input_len].numpy()           # (L, F)
        y = series[i + input_len : i + input_len + horizon].numpy()

        if kind == "naive":
            p = np.repeat(x[-1:, :], horizon, axis=0)   # (H, F)

        elif kind == "drift":
            if len(x) < 2:
                p = np.repeat(x[-1:, :], horizon, axis=0)
            else:
                slope = (x[-1] - x[0]) / (len(x) - 1)  # (F,)
                p = x[-1][None] + slope[None] * np.arange(1, horizon + 1)[:, None]

        else:
            raise ValueError(f"kind desconhecido: {kind}")

        preds.append(p)
        trues.append(y)

    return np.array(preds), np.array(trues)


def rolling_evaluate_fft_only(
    fft_pipeline_: FFTLSTMPipeline,
    series: torch.Tensor,
    *,
    input_len: int,
    horizon: int,
    test_start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrapolação FFT pura (sem LSTM) — baseline determinístico."""
    T_total = len(series)
    first_w = max(0, test_start - input_len)
    last_w  = T_total - input_len - horizon
    preds, trues = [], []

    for i in range(first_w, last_w + 1):
        y    = series[i + input_len : i + input_len + horizon].numpy()
        pred = fft_pipeline_.fft_extrapolation(
            start=i + input_len, horizon=horizon
        )                                                 # (H, F)
        preds.append(pred)
        trues.append(y)

    return np.array(preds), np.array(trues)


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict[str, float]:
    """Agrega métricas sobre todas as janelas e canais."""
    y_hat = preds.reshape(-1)
    y_tru = trues.reshape(-1)
    return {
        "MAE":   float(np_mae(y_hat, y_tru)),
        "RMSE":  float(np_rmse(y_hat, y_tru)),
        "sMAPE": float(np_smape(y_hat, y_tru)),
    }


# Carregar melhor checkpoint salvo pelo Trainer (chave "model_state")
def _load_best(model: nn.Module, result: dict) -> None:
    path = result.get("best_path")
    if path and Path(path).exists():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Melhor checkpoint carregado de: {path}")

_load_best(hybrid_model, hybrid_result)
_load_best(lstm_model,   lstm_result)

hybrid_model.to(DEVICE)
lstm_model.to(DEVICE)

print("\n=== [11] Avaliação rolling no conjunto de teste ===")
print(f"  Protocolo: janelas deslizantes | input={INPUT_LEN} | horizon={HORIZON}")
print(f"  Test start idx: {n_train + n_val} (step {n_train + n_val + 1} da série)")

# Naive
preds_naive, trues_naive = rolling_evaluate_naive(
    series_tensor, input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, kind="naive"
)

# Drift
preds_drift, trues_drift = rolling_evaluate_naive(
    series_tensor, input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, kind="drift"
)

# FFT-only
preds_fft, trues_fft = rolling_evaluate_fft_only(
    fft_pipeline, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON, test_start=n_train + n_val
)

# LSTM puro
preds_lstm, trues_lstm = rolling_evaluate_model(
    lstm_model, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, device=DEVICE, mode="raw"
)

# Híbrido FFT+LSTM
preds_hybrid, trues_hybrid = rolling_evaluate_model(
    hybrid_model, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, device=DEVICE,
    mode="residual", fft_pipeline_=fft_pipeline
)

metrics = {
    "Naive":      compute_metrics(preds_naive,  trues_naive),
    "Drift":      compute_metrics(preds_drift,  trues_drift),
    "FFT-only":   compute_metrics(preds_fft,    trues_fft),
    "LSTM puro":  compute_metrics(preds_lstm,   trues_lstm),
    "FFT+LSTM":   compute_metrics(preds_hybrid, trues_hybrid),
}


# ============================================================
# 12) Tabela de resultados
# ============================================================
print("\n" + "=" * 62)
print("RESULTADOS NO CONJUNTO DE TESTE (holdout nunca visto no treino)")
print("=" * 62)
print(f"{'Modelo':<14}  {'MAE':>8}  {'RMSE':>8}  {'sMAPE%':>9}")
print("-" * 62)

# Ordenar por MAE
ranked = sorted(metrics.items(), key=lambda kv: kv[1]["MAE"])
for name, m in ranked:
    print(f"{name:<14}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {m['sMAPE']:>9.2f}")

print("=" * 62)

# Ganho do híbrido sobre o LSTM puro
mae_lstm   = metrics["LSTM puro"]["MAE"]
mae_hybrid = metrics["FFT+LSTM"]["MAE"]
if mae_lstm > 0:
    ganho = (mae_lstm - mae_hybrid) / mae_lstm * 100
    print(f"\nGanho do FFT+LSTM sobre LSTM puro: {ganho:+.1f}% (MAE)")

# Nota sobre realismo
print("""
Nota sobre os resultados:
  • O 'ruído irredutível' (piso de Bayes) deste dataset é ≈ 0.25–0.35 (MAE
    normalizado). Qualquer modelo com MAE < 0.20 estaria overfittando.
  • O FFT captura tendência + sazonalidade; o LSTM aprende a estrutura
    remanescente do resíduo estocástico.
  • Nem o LSTM nem o híbrido são 'perfeitos' — isso seria ruído miraculoso.
  • Diferenças de MAE entre modelos refletem capacidade de generalização,
    não ajuste perfeito aos dados de teste.
""")


# ============================================================
# 13) (Opcional) Plot dos resultados — só executa se matplotlib
#     estiver disponível
# ============================================================
try:
    import matplotlib.pyplot as plt

    print("=== [13] Gerando plots de comparação ===")

    # Plota para o canal 0 (y1), 10 primeiras janelas de teste, horizonte completo
    channel = 0
    n_plot  = min(10, preds_naive.shape[0])

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]

    colors = {
        "Real":      "black",
        "Naive":     "gray",
        "FFT-only":  "steelblue",
        "LSTM puro": "orange",
        "FFT+LSTM":  "green",
    }

    for wi in range(n_plot):
        ax = axes[wi]
        true_w = trues_hybrid[wi, :, channel]
        h_range = np.arange(len(true_w))

        ax.plot(h_range, true_w,                        color=colors["Real"],
                linewidth=2.0, label="Real", zorder=5)
        ax.plot(h_range, preds_naive[wi, :, channel],   color=colors["Naive"],
                linestyle="--", alpha=0.6, label="Naive")
        ax.plot(h_range, preds_fft[wi, :, channel],     color=colors["FFT-only"],
                linestyle="-.", alpha=0.8, label="FFT-only")
        ax.plot(h_range, preds_lstm[wi, :, channel],    color=colors["LSTM puro"],
                linestyle=":",  alpha=0.8, label="LSTM puro")
        ax.plot(h_range, preds_hybrid[wi, :, channel],  color=colors["FFT+LSTM"],
                linewidth=1.5, alpha=0.9, label="FFT+LSTM")

        ax.set_ylabel("y1 (normalizado)")
        ax.set_title(f"Janela de teste #{wi + 1}")
        if wi == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=5)

    axes[-1].set_xlabel("Passo futuro (horizonte)")
    plt.suptitle(
        "Previsão por janela de teste — comparação de modelos\n"
        "(escala normalizada: média≈0, std≈1)",
        y=1.01, fontsize=11,
    )
    plt.tight_layout()

    plot_path = Path("outputs") / "07_forecast_comparison.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"  Plot salvo em: {plot_path.resolve()}")
    plt.close()

    # Segundo plot: espectro de potência (canal 0)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sp0 = spectra[0]
    mask_finite = np.isfinite(sp0["periods"]) & (sp0["periods"] < 200)
    ax2.bar(sp0["periods"][mask_finite], sp0["power"][mask_finite],
            width=0.8, alpha=0.7, color="steelblue", label="Potência FFT (y1)")
    ax2.set_xlabel("Período (steps)")
    ax2.set_ylabel("Potência")
    ax2.set_title("Espectro de Potência — canal y1 (dados de treino)")
    ax2.set_xlim(0, 100)
    ax2.legend()
    plt.tight_layout()

    spectrum_path = Path("outputs") / "07_power_spectrum.png"
    plt.savefig(spectrum_path, dpi=120, bbox_inches="tight")
    print(f"  Espectro salvo em: {spectrum_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib não disponível — plots pulados)")

print("\nPipeline concluído com sucesso.")
