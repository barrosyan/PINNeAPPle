"""Outlier detection and treatment for time series."""
from __future__ import annotations
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

TreatmentMethod = Literal["clip", "remove", "interpolate", "winsorize", "flag", "mean_replace"]
DetectionMethod = Literal["rolling_iqr", "zscore", "modified_zscore", "both"]


class OutlierDetector:
    """
    Outlier detection (Rolling IQR / Z-Score) with 6 treatment options.

    Detection methods:
      rolling_iqr    — IQR computed in a rolling window; flags points outside
                       [Q1 - iqr_factor*IQR, Q3 + iqr_factor*IQR]
      zscore         — flags |z| > threshold using rolling mean/std
      modified_zscore — MAD-based robust Z-score (Iglewicz-Hoaglin)
      both           — union of rolling_iqr and zscore flags

    Treatment options:
      clip         — clamp to [lower_fence, upper_fence]
      remove       — replace with NaN (use imputer afterward)
      interpolate  — linear interpolation over flagged points
      winsorize    — same as clip but uses percentile-based bounds
      flag         — no change, just return mask (call treat() explicitly)
      mean_replace — replace with rolling window mean
    """

    def __init__(
        self,
        method: DetectionMethod = "rolling_iqr",
        window: int = 20,
        iqr_factor: float = 1.5,
        z_threshold: float = 3.0,
        treatment: TreatmentMethod = "interpolate",
        winsorize_pct: float = 0.05,
    ):
        self.method = method
        self.window = int(window)
        self.iqr_factor = float(iqr_factor)
        self.z_threshold = float(z_threshold)
        self.treatment = treatment
        self.winsorize_pct = float(winsorize_pct)

    # ------------------------------------------------------------------
    def detect(self, s: pd.Series) -> pd.Series:
        """Return boolean Series (True = outlier)."""
        if self.method == "rolling_iqr":
            return self._rolling_iqr(s)
        if self.method == "zscore":
            return self._zscore(s)
        if self.method == "modified_zscore":
            return self._modified_zscore(s)
        if self.method == "both":
            return self._rolling_iqr(s) | self._zscore(s)
        raise ValueError(f"Unknown method: {self.method}")

    def treat(self, s: pd.Series, mask: pd.Series) -> pd.Series:
        """Apply treatment to flagged points."""
        out = s.copy()
        if not mask.any():
            return out
        if self.treatment == "clip":
            q1 = s.rolling(self.window, min_periods=3, center=True).quantile(0.25)
            q3 = s.rolling(self.window, min_periods=3, center=True).quantile(0.75)
            iqr = q3 - q1
            lo = q1 - self.iqr_factor * iqr
            hi = q3 + self.iqr_factor * iqr
            out = out.clip(lower=lo, upper=hi)
        elif self.treatment == "remove":
            out[mask] = np.nan
        elif self.treatment == "interpolate":
            out[mask] = np.nan
            out = out.interpolate(method="linear", limit_direction="both")
        elif self.treatment == "winsorize":
            lo_p = s.quantile(self.winsorize_pct)
            hi_p = s.quantile(1 - self.winsorize_pct)
            out = out.clip(lo_p, hi_p)
        elif self.treatment == "flag":
            pass  # user handles externally
        elif self.treatment == "mean_replace":
            roll_mean = s.rolling(self.window, min_periods=1, center=True).mean()
            out[mask] = roll_mean[mask]
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect and treat outliers in all target columns.

        Returns
        -------
        df_clean  : treated DataFrame
        flag_df   : boolean DataFrame of outlier flags (same shape as df[target_cols])
        """
        out = df.copy()
        if target_cols is None:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        flags = pd.DataFrame(False, index=df.index, columns=target_cols)
        for col in target_cols:
            mask = self.detect(out[col])
            flags[col] = mask
            out[col] = self.treat(out[col], mask)

        return out, flags

    # ------------------------------------------------------------------
    def _rolling_iqr(self, s: pd.Series) -> pd.Series:
        w = self.window
        q1 = s.rolling(w, min_periods=3, center=True).quantile(0.25)
        q3 = s.rolling(w, min_periods=3, center=True).quantile(0.75)
        iqr = (q3 - q1).clip(lower=1e-10)
        lo = q1 - self.iqr_factor * iqr
        hi = q3 + self.iqr_factor * iqr
        return (s < lo) | (s > hi)

    def _zscore(self, s: pd.Series) -> pd.Series:
        mu = s.rolling(self.window, min_periods=3, center=True).mean()
        sigma = s.rolling(self.window, min_periods=3, center=True).std().clip(lower=1e-10)
        z = (s - mu) / sigma
        return z.abs() > self.z_threshold

    def _modified_zscore(self, s: pd.Series) -> pd.Series:
        median = s.rolling(self.window, min_periods=3, center=True).median()
        mad = (s - median).abs().rolling(self.window, min_periods=3, center=True).median()
        mad = mad.clip(lower=1e-10)
        mz = 0.6745 * (s - median) / mad
        return mz.abs() > self.z_threshold

    def outlier_summary(self, flags: pd.DataFrame) -> pd.DataFrame:
        counts = flags.sum().rename("n_outliers")
        rates = (flags.mean() * 100).rename("outlier_pct")
        return pd.concat([counts, rates], axis=1)
