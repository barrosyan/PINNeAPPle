"""CSV → tensor loader for time series datasets.

Supports:
  - Any CSV with 1 to N numerical columns (univariate or multivariate)
  - Optional datetime column for sorting
  - Missing-value imputation (forward-fill, back-fill, or linear interpolation)
  - Channel-wise z-score normalisation fitted on the train split only
    (no leakage into val/test)
  - Automatic train/val/test temporal splits

Typical usage::

    from pinneaple_timeseries.csv_loader import load_timeseries_csv

    result = load_timeseries_csv(
        "data/temperature.csv",
        target_cols=["T_indoor", "T_outdoor"],
        time_col="timestamp",
        normalize=True,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    train_tensor = result["train_tensor"]   # (T_train, F) float32
    val_tensor   = result["val_tensor"]     # (T_val,   F)
    test_tensor  = result["test_tensor"]    # (T_test,  F)
    scaler       = result["scaler"]         # TimeSeriesScaler for inverse-transform
    meta         = result["meta"]           # dict with shapes, column names, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Channel-wise z-score scaler
# ---------------------------------------------------------------------------

class TimeSeriesScaler:
    """Channel-wise z-score scaler: fit on training data, apply to all splits.

    Avoids data leakage by only computing mean/std from the training portion.
    Stores running mean and std so any slice can be (inverse-)transformed later.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None   # (F,)
        self.std_:  Optional[np.ndarray] = None   # (F,)

    # ------------------------------------------------------------------
    def fit(self, y: np.ndarray) -> "TimeSeriesScaler":
        """Compute channel statistics from training data.  y: (T, F) or (T,)."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        self.mean_ = y.mean(axis=0)                       # (F,)
        self.std_  = y.std(axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Standardise y using fitted statistics.  y: (T, F) or (T,)."""
        self._check_fitted()
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 1
        if scalar:
            y = y[:, None]
        out = (y - self.mean_) / self.std_
        return out[:, 0] if scalar else out

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Revert standardisation.  y: (T, F) or (T,)."""
        self._check_fitted()
        y = np.asarray(y, dtype=float)
        scalar = y.ndim == 1
        if scalar:
            y = y[:, None]
        out = y * self.std_ + self.mean_
        return out[:, 0] if scalar else out

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("TimeSeriesScaler: call fit() before transform().")


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_timeseries_csv(
    path: Union[str, Path],
    *,
    target_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    time_col: Optional[str] = None,
    fill_method: str = "ffill",
    normalize: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict:
    """Load a CSV file and return train / val / test tensors ready for PINNeAPPle.

    The split is strictly temporal (no shuffling) to prevent look-ahead leakage.
    Normalisation statistics are fitted on the training split only.

    Args:
        path:        Path to the CSV file.
        target_cols: Columns to include as model targets.  If None, all numeric
                     columns (excluding time_col) are used.
        feature_cols: Additional exogenous feature columns to append.  If None,
                      no extra features are added beyond target_cols.
        time_col:    Optional column name containing timestamps.  When provided
                     the DataFrame is sorted chronologically before splitting.
        fill_method: Strategy for missing values: ``"ffill"`` (forward-fill),
                     ``"bfill"`` (back-fill), or ``"linear"`` (interpolation).
        normalize:   If True, apply channel-wise z-score normalisation fitted
                     on the training split.
        train_ratio: Fraction of time steps for training (default 0.70).
        val_ratio:   Fraction of time steps for validation (default 0.15).
                     The remainder  ``1 - train_ratio - val_ratio``  becomes the
                     test set.

    Returns:
        A dict with the following keys:

        ``series_tensor``  (T, F) float32 — full normalised series.
        ``train_tensor``   (T_train, F) float32
        ``val_tensor``     (T_val,   F) float32
        ``test_tensor``    (T_test,  F) float32
        ``scaler``         :class:`TimeSeriesScaler` or ``None`` when normalize=False.
        ``meta``           dict — shape info, column names, split sizes.
        ``raw_df``         pandas DataFrame before normalisation (original values).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for load_timeseries_csv.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Sort chronologically when a time column is present
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

    # Collect numeric columns (excluding the time column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if time_col and time_col in numeric_cols:
        numeric_cols.remove(time_col)

    if not numeric_cols:
        raise ValueError("No numeric columns found in the CSV.")

    # Determine which columns to use
    if target_cols is None:
        target_cols = numeric_cols
    else:
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"target_cols not found in CSV: {missing}")

    if feature_cols is None:
        feature_cols = []
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not found in CSV: {missing}")

    # Deduplicated ordered column list: targets first, then extra features
    all_cols: List[str] = list(dict.fromkeys(target_cols + feature_cols))
    df_sel = df[all_cols].copy()

    # --- Impute missing values ---
    if fill_method == "ffill":
        df_sel = df_sel.ffill().bfill()
    elif fill_method == "bfill":
        df_sel = df_sel.bfill().ffill()
    elif fill_method == "linear":
        df_sel = df_sel.interpolate(method="linear").ffill().bfill()
    else:
        raise ValueError(f"Unknown fill_method '{fill_method}'. Use 'ffill', 'bfill', or 'linear'.")

    raw_df = df_sel.copy()  # keep unscaled copy for reference
    data = df_sel.values.astype(np.float64)  # (T, F)
    T, F = data.shape

    # Temporal split indices
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)
    n_test  = T - n_train - n_val
    if n_test <= 0:
        raise ValueError(
            f"train_ratio + val_ratio = {train_ratio + val_ratio:.2f} leaves no test data "
            f"(T={T}). Reduce one of the ratios."
        )

    train_data = data[:n_train].copy()
    val_data   = data[n_train : n_train + n_val].copy()
    test_data  = data[n_train + n_val :].copy()

    # Normalise: fit ONLY on train, apply to all
    scaler: Optional[TimeSeriesScaler] = None
    if normalize:
        scaler = TimeSeriesScaler()
        train_data = scaler.fit_transform(train_data)
        val_data   = scaler.transform(val_data)
        test_data  = scaler.transform(test_data)
        full_data  = scaler.transform(data)
    else:
        full_data = data.copy()

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    meta = {
        "path":        str(path),
        "T":           T,
        "F":           F,
        "columns":     all_cols,
        "target_cols": target_cols,
        "feature_cols": feature_cols,
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      n_test,
        "normalized":  normalize,
        "fill_method": fill_method,
    }

    return {
        "series_tensor": _t(full_data),
        "train_tensor":  _t(train_data),
        "val_tensor":    _t(val_data),
        "test_tensor":   _t(test_data),
        "scaler":        scaler,
        "meta":          meta,
        "raw_df":        raw_df,
    }


# ---------------------------------------------------------------------------
# Convenience: generate and save a synthetic multivariate CSV for testing
# ---------------------------------------------------------------------------

def generate_synthetic_csv(
    path: Union[str, Path],
    T: int = 1000,
    seed: int = 42,
    freq: str = "D",
) -> Path:
    """Generate a synthetic multivariate time series CSV and save it to *path*.

    Creates three channels:
      - ``y1``: linear trend + two seasonal components + noise (main target)
      - ``y2``: different seasonality + weak cross-correlation with y1
      - ``y3``: weak trend + single seasonal component + higher noise

    Args:
        path:  Destination CSV file path.
        T:     Number of time steps.
        seed:  Random seed for reproducibility.
        freq:  Pandas frequency string used to build the datetime index.

    Returns:
        The resolved path of the written file.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for generate_synthetic_csv.") from exc

    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)

    # y1: upward trend + multi-period seasonality + moderate noise
    y1 = (
        0.005 * t
        + 2.0  * np.sin(2 * np.pi * t / 50.0)   # long season
        + 0.8  * np.sin(2 * np.pi * t / 20.0)   # medium season
        + 0.4  * np.sin(2 * np.pi * t / 7.0)    # weekly cycle
        + 0.3  * rng.standard_normal(T)
    )

    # y2: different phase / period + weak coupling to y1
    y2 = (
        0.002 * t
        + 1.5  * np.cos(2 * np.pi * t / 50.0)
        + 0.6  * np.sin(2 * np.pi * t / 25.0)
        + 0.15 * y1                              # weak cross-correlation
        + 0.4  * rng.standard_normal(T)
    )

    # y3: noisy signal with a single weaker season
    y3 = (
        0.001 * t
        + 0.8  * np.sin(2 * np.pi * t / 30.0)
        + 0.7  * rng.standard_normal(T)
    )

    # Inject a small number of NaN values to exercise the imputer
    for arr in (y1, y2, y3):
        missing_idx = rng.integers(50, T - 50, size=5)
        arr[missing_idx] = np.nan

    dates = pd.date_range(start="2020-01-01", periods=T, freq=freq)
    df = pd.DataFrame({"date": dates, "y1": y1, "y2": y2, "y3": y3})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
