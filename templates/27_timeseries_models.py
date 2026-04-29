"""27_timeseries_models.py — Classical and neural time-series models.

Demonstrates:
- ARIMAModel: auto-regressive integrated moving average
- ExponentialSmoothing: Holt-Winters triple exponential smoothing
- TCNModel: Temporal Convolutional Network for sequence regression
- InformerModel: Transformer-based long-sequence forecasting
- Multi-model comparison on a synthetic seasonal load curve
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_timeseries.arima import ARIMAModel
from pinneaple_timeseries.exponential_smoothing import ExponentialSmoothing
from pinneaple_models.recurrent.tcn import TCNModel

try:
    from pinneaple_models.transformers.informer import InformerModel
    _INFORMER = True
except ImportError:
    _INFORMER = False


# ---------------------------------------------------------------------------
# Synthetic dataset: daily load curve with trend + seasonality + noise
# ---------------------------------------------------------------------------

N_POINTS  = 500
PERIOD    = 24        # 24-hour seasonality (hourly data)
LOOK_BACK = 48        # history window for neural models
HORIZON   = 24        # forecast horizon


def generate_load_curve(n: int = N_POINTS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = np.arange(n)
    trend    = 0.01 * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / PERIOD) + \
               2.0 * np.cos(4 * np.pi * t / PERIOD)
    noise    = rng.normal(0, 0.5, n)
    return (trend + seasonal + noise + 20.0).astype(np.float32)


def make_sequences(series: np.ndarray, look_back: int, horizon: int):
    X, Y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i: i + look_back])
        Y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    series = generate_load_curve(N_POINTS)
    n_train_ts = int(0.8 * N_POINTS)
    train_series = series[:n_train_ts]
    test_series  = series[n_train_ts:]

    results = {}

    # =========================================================================
    # 1) ARIMA
    # =========================================================================
    print("\n[1] ARIMA ...")
    arima = ARIMAModel(order=(2, 1, 2), seasonal_order=(1, 1, 1, PERIOD))
    arima.fit(train_series)
    arima_fc = arima.forecast(steps=len(test_series))
    arima_rmse = float(np.sqrt(((arima_fc - test_series)**2).mean()))
    print(f"  ARIMA RMSE = {arima_rmse:.4f}")
    results["ARIMA"] = arima_fc

    # =========================================================================
    # 2) Exponential Smoothing (Holt-Winters)
    # =========================================================================
    print("[2] Holt-Winters ...")
    hw = ExponentialSmoothing(
        trend="add", seasonal="add", seasonal_periods=PERIOD
    )
    hw.fit(train_series)
    hw_fc = hw.forecast(steps=len(test_series))
    hw_rmse = float(np.sqrt(((hw_fc - test_series)**2).mean()))
    print(f"  HW RMSE = {hw_rmse:.4f}")
    results["Holt-Winters"] = hw_fc

    # =========================================================================
    # 3) TCN (Temporal Convolutional Network)
    # =========================================================================
    print("[3] TCN ...")
    X_all, Y_all = make_sequences(series, LOOK_BACK, HORIZON)
    n_train_seq  = n_train_ts - LOOK_BACK - HORIZON + 1
    X_tr = torch.tensor(X_all[:n_train_seq, :, None], device=device)   # (N, T, 1)
    Y_tr = torch.tensor(Y_all[:n_train_seq], device=device)             # (N, H)
    X_te = torch.tensor(X_all[n_train_seq:, :, None], device=device)
    Y_te = Y_all[n_train_seq:]

    tcn = TCNModel(
        in_channels=1,
        out_channels=HORIZON,
        n_levels=4,
        kernel_size=3,
        hidden_channels=32,
        dropout=0.1,
    ).to(device)

    opt_tcn = torch.optim.Adam(tcn.parameters(), lr=1e-3)
    for ep in range(1, 101):
        idx = torch.randperm(len(X_tr), device=device)
        for i in range(0, len(X_tr), 64):
            bi = idx[i: i + 64]
            opt_tcn.zero_grad()
            y_hat = tcn(X_tr[bi])
            (y_hat - Y_tr[bi]).pow(2).mean().backward()
            opt_tcn.step()
    tcn.eval()
    with torch.no_grad():
        tcn_pred = tcn(X_te).cpu().numpy()
    tcn_rmse = float(np.sqrt(((tcn_pred - Y_te)**2).mean()))
    print(f"  TCN RMSE = {tcn_rmse:.4f}")
    # Flatten predictions for single-step comparison
    results["TCN"] = tcn_pred[:, 0]

    # =========================================================================
    # 4) Informer (optional)
    # =========================================================================
    if _INFORMER:
        print("[4] Informer ...")
        informer = InformerModel(
            seq_len=LOOK_BACK,
            label_len=LOOK_BACK // 2,
            pred_len=HORIZON,
            d_model=32,
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=64,
            dropout=0.1,
            factor=5,
            output_attention=False,
        ).to(device)

        opt_inf = torch.optim.Adam(informer.parameters(), lr=5e-4)
        for ep in range(1, 51):
            for i in range(0, len(X_tr), 64):
                opt_inf.zero_grad()
                enc_in = X_tr[i: i + 64]               # (bs, T, 1)
                dec_in = enc_in[:, -LOOK_BACK // 2:, :]
                y_hat  = informer(enc_in, dec_in)
                Y_batch = Y_tr[i: i + 64].unsqueeze(-1) # (bs, H, 1)
                (y_hat - Y_batch).pow(2).mean().backward()
                opt_inf.step()
        informer.eval()
        with torch.no_grad():
            dec_te = X_te[:, -LOOK_BACK // 2:, :]
            inf_pred = informer(X_te, dec_te).squeeze(-1).cpu().numpy()
        inf_rmse = float(np.sqrt(((inf_pred - Y_te)**2).mean()))
        print(f"  Informer RMSE = {inf_rmse:.4f}")
        results["Informer"] = inf_pred[:, 0]

    # =========================================================================
    # Visualisation
    # =========================================================================
    t_all  = np.arange(N_POINTS)
    t_test = np.arange(n_train_ts, N_POINTS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(t_all, series, "k-", alpha=0.5, label="Full series")
    axes[0].axvline(n_train_ts, color="gray", ls="--", label="Train/Test split")
    for label, fc in results.items():
        n_fc = min(len(fc), len(t_test))
        axes[0].plot(t_test[:n_fc], fc[:n_fc], label=label)
    axes[0].set_title("Time-series forecasting comparison")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Load")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Zoom on test window
    axes[1].plot(t_test[:72], test_series[:72], "k-", label="Truth", lw=2)
    for label, fc in results.items():
        n_fc = min(len(fc), 72)
        axes[1].plot(t_test[:n_fc], fc[:n_fc], label=label, alpha=0.8)
    axes[1].set_title("Forecast detail (first 72 test hours)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    rmse_labels = [f"ARIMA={arima_rmse:.2f}", f"HW={hw_rmse:.2f}",
                   f"TCN={tcn_rmse:.2f}"]
    axes[1].set_ylabel("Load  |  RMSEs: " + "  ".join(rmse_labels))

    plt.tight_layout()
    plt.savefig("27_timeseries_models_result.png", dpi=120)
    print("Saved 27_timeseries_models_result.png")


if __name__ == "__main__":
    main()
