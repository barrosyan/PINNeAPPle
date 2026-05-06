"""Missing value treatment with layered interpolation strategies."""
from __future__ import annotations
from typing import List, Optional, Union

import numpy as np
import pandas as pd


_STRATEGIES = ("linear", "pchip", "forward_fill", "backward_fill", "mean", "zero", "layered")


class TimeSeriesImputer:
    """
    Layered missing-value imputer for time series.

    strategy="layered" applies:
      1. Linear interpolation (handles interior gaps)
      2. PCHIP interpolation (smooth, monotone-preserving)
      3. Forward-fill (propagates last valid observation)
      4. Backward-fill (fills leading NaNs)
      5. Zero-fill (last resort)

    All other strategies apply that single method directly.
    """

    def __init__(
        self,
        strategy: str = "layered",
        max_gap: Optional[int] = None,
        verbose: bool = False,
    ):
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}")
        self.strategy = strategy
        self.max_gap = max_gap
        self.verbose = verbose
        self._nan_counts: dict = {}

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Impute missing values in-place (returns a copy).

        Parameters
        ----------
        df          : input DataFrame (must be sorted by time if time_col given)
        target_cols : columns to impute; None → all numeric columns
        time_col    : datetime index column (used only to sort)
        """
        out = df.copy()
        if time_col and time_col in out.columns:
            out = out.sort_values(time_col).reset_index(drop=True)

        if target_cols is None:
            target_cols = out.select_dtypes(include=[np.number]).columns.tolist()

        for col in target_cols:
            s = out[col].copy()
            n_before = s.isna().sum()
            if n_before == 0:
                continue
            if self.verbose:
                print(f"[imputer] {col}: {n_before} missing values")

            if self.strategy == "layered":
                s = self._layered(s)
            else:
                s = self._single(s, self.strategy)

            self._nan_counts[col] = n_before
            out[col] = s
        return out

    # ------------------------------------------------------------------
    def _layered(self, s: pd.Series) -> pd.Series:
        s = self._single(s, "linear")
        if s.isna().any():
            s = self._single(s, "pchip")
        if s.isna().any():
            s = self._single(s, "forward_fill")
        if s.isna().any():
            s = self._single(s, "backward_fill")
        if s.isna().any():
            s = s.fillna(0.0)
        return s

    def _single(self, s: pd.Series, method: str) -> pd.Series:
        if method == "linear":
            return s.interpolate(method="linear", limit=self.max_gap, limit_direction="both")
        if method == "pchip":
            try:
                from scipy.interpolate import PchipInterpolator
                valid = s.dropna()
                if len(valid) < 2:
                    return s
                xi = valid.index.to_numpy(dtype=float)
                yi = valid.values
                interp = PchipInterpolator(xi, yi, extrapolate=False)
                missing = s.index[s.isna()].to_numpy(dtype=float)
                filled = interp(missing)
                s = s.copy()
                s.iloc[s.isna().to_numpy()] = filled
                return s
            except ImportError:
                return s.interpolate(method="polynomial", order=3, limit=self.max_gap)
        if method == "forward_fill":
            return s.ffill(limit=self.max_gap)
        if method == "backward_fill":
            return s.bfill(limit=self.max_gap)
        if method == "mean":
            return s.fillna(s.mean())
        if method == "zero":
            return s.fillna(0.0)
        return s

    @property
    def nan_report(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(self._nan_counts.items()), columns=["column", "n_missing"]
        )
