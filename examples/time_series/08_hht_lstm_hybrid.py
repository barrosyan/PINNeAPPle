"""
08_hht_lstm_hybrid.py — Hybrid HHT (Hilbert-Huang Transform) + LSTM Residual Forecasting

Pipeline overview
-----------------
0. Reproducibility: global seed fixed.
1. Generation and saving of a synthetic multivariate CSV dataset (same as example 07).
2. Loading the CSV via the PINNeAPPle csv_loader (works with ANY CSV).
3. HHT analysis: EMD decomposition and dominant IMF info per variable.
4. HHT decomposition fitted ONLY on the training set (no leakage).
5. Residual computation for the entire dataset.
6. Construction of TSDataModules for the hybrid model and for the pure LSTM.
7. Training with anti-overfitting strategies: dropout, weight_decay, grad_clip,
   EarlyStopping, and proper temporal split.
8. Rolling evaluation on the test set (holdout never seen during training).
9. Fair comparison between 4 approaches:
   • Naive (repeat last value)
   • HHT-only (deterministic extrapolation via Hilbert instantaneous frequency)
   • Pure LSTM (trained on normalized raw data)
   • Hybrid HHT + LSTM (trained on residuals after HHT decomposition)

## Why HHT vs FFT?

- FFT assumes stationarity and uses a global fixed-frequency basis.
- EMD (Empirical Mode Decomposition) is data-adaptive: it extracts Intrinsic Mode
  Functions (IMFs) directly from the signal, capturing non-stationary amplitude and
  frequency modulation.
- The Hilbert Transform then gives instantaneous frequency/amplitude for each IMF,
  allowing forward extrapolation.
- Trade-off: HHT is more flexible for non-stationary signals but less precise for
  long-horizon extrapolation compared to FFT on strongly periodic data.

## Realism philosophy (same as example 07)

Models are NOT magical. The stochastic component is irreducible. Expected metrics
reflect the Bayes floor imposed by the additive noise (≈0.25-0.35 normalized MAE).

Anti-leakage guarantees:
* TimeSeriesScaler fitted ONLY on the training set.
* HHTDecomposer fitted ONLY on the training set.
* EarlyStopping monitors val_total (not test).
* The test set is never seen during training or validation.

External dependencies: pandas, numpy, torch, PyEMD (pip install EMD-signal),
scipy, (optional) matplotlib.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneaple_timeseries.csv_loader import generate_synthetic_csv, load_timeseries_csv
from pinneaple_timeseries.decomposition.hht_lstm import HHTLSTMPipeline
from pinneaple_timeseries.datamodule import TSDataModule
from pinneaple_timeseries.spec import TimeSeriesSpec
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
# 0) Reproducibility
# ============================================================
SEED      = 42
INPUT_LEN = 96    # context window for the LSTM
HORIZON   = 24    # forecast horizon
BATCH     = 64
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(f"Dispositivo: {DEVICE}")


# ============================================================
# 1) Generate and save synthetic CSV
# ============================================================
CSV_PATH = Path("data") / "synthetic_multivariate.csv"

print("\n=== [1] Gerando dataset CSV sintetico ===")
if not CSV_PATH.exists():
    generate_synthetic_csv(CSV_PATH, T=1000, seed=SEED, freq="D")
    print(f"CSV salvo em: {CSV_PATH.resolve()}")
else:
    print(f"CSV ja existe: {CSV_PATH.resolve()} (reutilizando)")


# ============================================================
# 2) Load CSV
# ============================================================
print("\n=== [2] Carregando CSV ===")
result = load_timeseries_csv(
    CSV_PATH,
    target_cols=["y1", "y2", "y3"],
    time_col="date",
    fill_method="linear",
    normalize=True,
    train_ratio=0.70,
    val_ratio=0.15,
)

series_tensor = result["series_tensor"]   # (1000, 3)
train_tensor  = result["train_tensor"]    # (700, 3)
val_tensor    = result["val_tensor"]      # (150, 3)
test_tensor   = result["test_tensor"]     # (150, 3)
scaler        = result["scaler"]
meta          = result["meta"]

T, F = meta["T"], meta["F"]
n_train, n_val, n_test = meta["n_train"], meta["n_val"], meta["n_test"]

print(f"Serie: T={T} | F={F} colunas: {meta['columns']}")
print(f"Splits: treino={n_train} | val={n_val} | teste={n_test} steps")
print(f"Normalizado: {meta['normalized']}")


# ============================================================
# 3) HHT analysis — EMD on training data, show IMF info
# ============================================================
print("\n=== [3] Analise HHT dos dados de treino ===")

hht_pipeline = HHTLSTMPipeline(
    n_imfs=6,
    dominant_energy_ratio=0.80,
    detrend=True,
    input_len=INPUT_LEN,
    horizon=HORIZON,
    n_features=F,
    n_targets=F,
    hidden_size=128,
    num_layers=2,
    dropout=0.20,
)

# Fit HHT on training data only — also populates IMF info
hht_pipeline.fit_hht(train_tensor.numpy())   # (700, 3) — training only
imf_infos = hht_pipeline.imf_info()

for col, info in zip(meta["columns"], imf_infos):
    n_d = info["n_dominant"]
    top_periods = sorted(info["periods"])[:3]
    periods_str = ", ".join(f"{p:.1f}" for p in top_periods if math.isfinite(p))
    print(f"  Canal '{col}': {n_d} IMFs dominantes | periodos aprox. [{periods_str}] steps")


# ============================================================
# 4) HHTDecomposer already fitted in step 3 — report dominant counts
# ============================================================
print("\n=== [4] HHTDecomposer ajustado no treino ===")
counts = hht_pipeline.dominant_imf_counts()
for col, cnt in zip(meta["columns"], counts):
    print(f"  '{col}': {cnt} IMFs dominantes selecionados")


# ============================================================
# 5) Compute residuals for the full series
#    (HHT reconstructs training, extrapolates val/test — no leakage)
# ============================================================
print("\n=== [5] Calculando residuos (original - HHT) ===")
residuals_tensor = hht_pipeline.residuals_tensor(series_tensor.numpy())  # (1000, 3)

res_train = residuals_tensor[:n_train]
res_val   = residuals_tensor[n_train : n_train + n_val]
res_test  = residuals_tensor[n_train + n_val :]

print(f"  Residuo treino  -- std por canal: {res_train.std(0).numpy().round(4)}")
print(f"  Residuo val     -- std por canal: {res_val.std(0).numpy().round(4)}")
print(f"  Serie original  -- std por canal: {train_tensor.std(0).numpy().round(4)}")


# ============================================================
# 6) TSDataModules — train+val only; test is held out
# ============================================================
spec = TimeSeriesSpec(input_len=INPUT_LEN, horizon=HORIZON, stride=1)

n_trainval = n_train + n_val
val_frac   = n_val / n_trainval

# HHT hybrid: LSTM trains on residuals
residuals_trainval = residuals_tensor[:n_trainval]
dm_hybrid = TSDataModule(
    series=residuals_trainval,
    spec=spec,
    batch_size=BATCH,
    val_ratio=val_frac,
    num_workers=0,
)

# Pure LSTM: trains on raw normalized series
raw_trainval = series_tensor[:n_trainval]
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
# 7) LSTM model definition (same anti-overfitting architecture as example 07)
# ============================================================

class _LSTM(nn.Module):
    """LSTM with anti-overfitting:
    LayerNorm input, inter-layer dropout, LayerNorm on hidden state, head dropout.
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
        xn = self.input_norm(x)
        out, _ = self.lstm(xn)
        last = self.post_norm(out[:, -1, :])
        last = self.head_drop(last)
        pred = self.head(last)
        return pred.view(x.shape[0], self.horizon, self.n_features)


def _build_lstm() -> _LSTM:
    return _LSTM(n_features=F, hidden_size=128, num_layers=2, dropout=0.20, horizon=HORIZON)


# ============================================================
# 8) Training configuration
# ============================================================

def _make_train_cfg(run_name: str) -> TrainConfig:
    return TrainConfig(
        epochs=150,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
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
            RobustScaleStep(key="x", enabled=True),
        ]),
        early_stopping=EarlyStopping(
            patience=15,
            monitor="val_total",
            mode="min",
        ),
        checkpoint=ModelCheckpoint(
            path=f"checkpoints/{run_name}.pt",
            monitor="val_total",
        ),
    )


# ============================================================
# 9) Train HYBRID model (HHT residuals -> LSTM)
# ============================================================
print("\n=== [9] Treinando modelo HIBRIDO HHT+LSTM ===")

hybrid_model   = _build_lstm()
hybrid_trainer = _make_trainer(hybrid_model, "hht_hybrid_lstm")
hybrid_result  = hybrid_trainer.fit(train_loader_h, val_loader_h, _make_train_cfg("hht_hybrid"))

print(f"  Melhor val_total: {hybrid_result.get('best_val', float('nan')):.6f}")
print(f"  Checkpoint: {hybrid_result.get('best_path', 'n/a')}")


# ============================================================
# 10) Train PURE LSTM baseline
# ============================================================
print("\n=== [10] Treinando LSTM PURO (baseline neural) ===")

lstm_model   = _build_lstm()
lstm_trainer = _make_trainer(lstm_model, "hht_pure_lstm")
lstm_result  = lstm_trainer.fit(train_loader_r, val_loader_r, _make_train_cfg("hht_lstm"))

print(f"  Melhor val_total: {lstm_result.get('best_val', float('nan')):.6f}")
print(f"  Checkpoint: {lstm_result.get('best_path', 'n/a')}")


# ============================================================
# 11) Load best checkpoints and evaluate on test set
# ============================================================

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

print("\n=== [11] Avaliacao rolling no conjunto de teste ===")
print(f"  Protocolo: janelas deslizantes | input={INPUT_LEN} | horizon={HORIZON}")
print(f"  Test start idx: {n_train + n_val} (step {n_train + n_val + 1} da serie)")


def rolling_evaluate_model(
    model: nn.Module,
    series: torch.Tensor,
    *,
    input_len: int,
    horizon: int,
    test_start: int,
    device: str,
    mode: str = "raw",
    hht_pipeline_: HHTLSTMPipeline | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    T_total = len(series)
    first_window = max(0, test_start - input_len)
    last_window  = T_total - input_len - horizon

    if first_window > last_window:
        raise ValueError("test_start ou horizon incompativeis com o tamanho da serie.")

    preds, trues = [], []
    with torch.no_grad():
        for i in range(first_window, last_window + 1):
            x_raw  = series[i : i + input_len]
            y_true = series[i + input_len : i + input_len + horizon]

            if mode == "residual":
                x_input = residuals_tensor[i : i + input_len]
                x_batch = x_input.unsqueeze(0).to(device)
                y_res   = model(x_batch).squeeze(0).cpu().numpy()

                hht_part = hht_pipeline_.hht_extrapolation(
                    start=i + input_len,
                    horizon=horizon,
                )
                y_pred = hht_part + y_res

            else:
                x_batch = x_raw.unsqueeze(0).to(device)
                y_pred  = model(x_batch).squeeze(0).cpu().numpy()

            preds.append(y_pred)
            trues.append(y_true.numpy())

    return np.array(preds), np.array(trues)


def rolling_evaluate_naive(
    series: torch.Tensor,
    *,
    input_len: int,
    horizon: int,
    test_start: int,
    kind: str = "naive",
) -> Tuple[np.ndarray, np.ndarray]:
    T_total = len(series)
    first_w = max(0, test_start - input_len)
    last_w  = T_total - input_len - horizon
    preds, trues = [], []

    for i in range(first_w, last_w + 1):
        x = series[i : i + input_len].numpy()
        y = series[i + input_len : i + input_len + horizon].numpy()

        if kind == "naive":
            p = np.repeat(x[-1:, :], horizon, axis=0)
        elif kind == "drift":
            if len(x) < 2:
                p = np.repeat(x[-1:, :], horizon, axis=0)
            else:
                slope = (x[-1] - x[0]) / (len(x) - 1)
                p = x[-1][None] + slope[None] * np.arange(1, horizon + 1)[:, None]
        else:
            raise ValueError(f"kind desconhecido: {kind}")

        preds.append(p)
        trues.append(y)

    return np.array(preds), np.array(trues)


def rolling_evaluate_hht_only(
    hht_pipeline_: HHTLSTMPipeline,
    series: torch.Tensor,
    *,
    input_len: int,
    horizon: int,
    test_start: int,
) -> Tuple[np.ndarray, np.ndarray]:
    T_total = len(series)
    first_w = max(0, test_start - input_len)
    last_w  = T_total - input_len - horizon
    preds, trues = [], []

    for i in range(first_w, last_w + 1):
        y    = series[i + input_len : i + input_len + horizon].numpy()
        pred = hht_pipeline_.hht_extrapolation(
            start=i + input_len, horizon=horizon
        )
        preds.append(pred)
        trues.append(y)

    return np.array(preds), np.array(trues)


def compute_metrics(preds: np.ndarray, trues: np.ndarray) -> Dict[str, float]:
    y_hat = preds.reshape(-1)
    y_tru = trues.reshape(-1)
    return {
        "MAE":   float(np_mae(y_hat, y_tru)),
        "RMSE":  float(np_rmse(y_hat, y_tru)),
        "sMAPE": float(np_smape(y_hat, y_tru)),
    }


# Evaluate all models
preds_naive, trues_naive = rolling_evaluate_naive(
    series_tensor, input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, kind="naive"
)
preds_drift, trues_drift = rolling_evaluate_naive(
    series_tensor, input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, kind="drift"
)
preds_hht, trues_hht = rolling_evaluate_hht_only(
    hht_pipeline, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON, test_start=n_train + n_val
)
preds_lstm, trues_lstm = rolling_evaluate_model(
    lstm_model, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, device=DEVICE, mode="raw"
)
preds_hybrid, trues_hybrid = rolling_evaluate_model(
    hybrid_model, series_tensor,
    input_len=INPUT_LEN, horizon=HORIZON,
    test_start=n_train + n_val, device=DEVICE,
    mode="residual", hht_pipeline_=hht_pipeline
)

metrics = {
    "Naive":      compute_metrics(preds_naive,  trues_naive),
    "Drift":      compute_metrics(preds_drift,  trues_drift),
    "HHT-only":   compute_metrics(preds_hht,    trues_hht),
    "LSTM puro":  compute_metrics(preds_lstm,   trues_lstm),
    "HHT+LSTM":   compute_metrics(preds_hybrid, trues_hybrid),
}


# ============================================================
# 12) Results table
# ============================================================
print("\n" + "=" * 62)
print("RESULTADOS NO CONJUNTO DE TESTE (holdout nunca visto no treino)")
print("=" * 62)
print(f"{'Modelo':<14}  {'MAE':>8}  {'RMSE':>8}  {'sMAPE%':>9}")
print("-" * 62)

ranked = sorted(metrics.items(), key=lambda kv: kv[1]["MAE"])
for name, m in ranked:
    print(f"{name:<14}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {m['sMAPE']:>9.2f}")

print("=" * 62)

mae_lstm   = metrics["LSTM puro"]["MAE"]
mae_hybrid = metrics["HHT+LSTM"]["MAE"]
if mae_lstm > 0:
    ganho = (mae_lstm - mae_hybrid) / mae_lstm * 100
    print(f"\nGanho do HHT+LSTM sobre LSTM puro: {ganho:+.1f}% (MAE)")

print("""
Nota sobre os resultados:
  - O 'ruido irredutivel' (piso de Bayes) deste dataset e aprox. 0.25-0.35 (MAE
    normalizado). Qualquer modelo com MAE < 0.20 estaria overfittando.
  - O HHT captura componentes nao-estacionarias; o LSTM aprende a estrutura
    remanescente do residuo estocastico.
  - A qualidade da extrapolacao HHT depende da estabilidade das IMFs no horizonte.
  - Diferencas de MAE entre modelos refletem capacidade de generalizacao.
""")


# ============================================================
# 13) (Optional) Plots
# ============================================================
try:
    import matplotlib.pyplot as plt

    print("=== [13] Gerando plots de comparacao ===")

    channel = 0
    n_plot  = min(10, preds_naive.shape[0])

    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=False)
    if n_plot == 1:
        axes = [axes]

    colors = {
        "Real":      "black",
        "Naive":     "gray",
        "HHT-only":  "steelblue",
        "LSTM puro": "orange",
        "HHT+LSTM":  "green",
    }

    for wi in range(n_plot):
        ax = axes[wi]
        true_w  = trues_hybrid[wi, :, channel]
        h_range = np.arange(len(true_w))

        ax.plot(h_range, true_w,                       color=colors["Real"],
                linewidth=2.0, label="Real", zorder=5)
        ax.plot(h_range, preds_naive[wi, :, channel],  color=colors["Naive"],
                linestyle="--", alpha=0.6, label="Naive")
        ax.plot(h_range, preds_hht[wi, :, channel],    color=colors["HHT-only"],
                linestyle="-.", alpha=0.8, label="HHT-only")
        ax.plot(h_range, preds_lstm[wi, :, channel],   color=colors["LSTM puro"],
                linestyle=":", alpha=0.8, label="LSTM puro")
        ax.plot(h_range, preds_hybrid[wi, :, channel], color=colors["HHT+LSTM"],
                linewidth=1.5, alpha=0.9, label="HHT+LSTM")

        ax.set_ylabel("y1 (normalizado)")
        ax.set_title(f"Janela de teste #{wi + 1}")
        if wi == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=5)

    axes[-1].set_xlabel("Passo futuro (horizonte)")
    plt.suptitle(
        "Previsao por janela de teste — comparacao de modelos (HHT)\n"
        "(escala normalizada: media~0, std~1)",
        y=1.01, fontsize=11,
    )
    plt.tight_layout()

    plot_path = Path("outputs") / "08_hht_forecast_comparison.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"  Plot salvo em: {plot_path.resolve()}")
    plt.close()

    # IMF amplitude plot per channel
    fig2, axes2 = plt.subplots(1, F, figsize=(5 * F, 4))
    if F == 1:
        axes2 = [axes2]

    for fi, (col, info) in enumerate(zip(meta["columns"], imf_infos)):
        ax = axes2[fi]
        amps    = info["amplitudes"]
        periods = [min(p, 500) for p in info["periods"]]
        bars = ax.bar(range(len(amps)), amps, color="teal", alpha=0.7)
        ax.set_xticks(range(len(amps)))
        ax.set_xticklabels([f"~{p:.0f}s" for p in periods], fontsize=8)
        ax.set_title(f"Canal {col}: amplitudes IMF dominantes")
        ax.set_xlabel("IMF (periodo aprox.)")
        ax.set_ylabel("Amplitude estimada")

    plt.suptitle("Amplitudes dos IMFs dominantes por canal (HHT)", fontsize=11)
    plt.tight_layout()

    imf_plot_path = Path("outputs") / "08_hht_imf_amplitudes.png"
    plt.savefig(imf_plot_path, dpi=120, bbox_inches="tight")
    print(f"  IMF plot salvo em: {imf_plot_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib nao disponivel -- plots pulados)")

print("\nPipeline concluido com sucesso.")
