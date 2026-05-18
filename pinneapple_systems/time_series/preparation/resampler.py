"""Automatic upsampling / downsampling for numeric time-indexed DataFrames."""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

AggMethod = Literal["mean", "median", "sum", "min", "max", "last", "first"]

_FREQ_ORDER = {
    "s": 0, "min": 1, "T": 1, "h": 2, "H": 2,
    "D": 3, "W": 4, "ME": 5, "M": 5, "QE": 6, "Q": 6, "YE": 7, "Y": 7,
}


def _parse_freq_rank(freq: str) -> int:
    base = "".join(c for c in freq if c.isalpha())
    return _FREQ_ORDER.get(base, -1)


class TimeSeriesResampler:
    """
    Resample a time-indexed DataFrame to a target frequency.

    Handles both:
      - Downsampling (e.g., seconds → minutes): aggregates via agg_method
      - Upsampling  (e.g., daily → hourly):     interpolates or forward-fills

    Parameters
    ----------
    target_freq : pandas offset alias (e.g. "1min", "1h", "1D")
    time_col    : name of the datetime column (or None if already the index)
    agg_method  : aggregation for downsampling
    upsample_method : "linear" | "pchip" | "ffill" | "bfill"
    numeric_only : only resample numeric columns
    """

    def __init__(
        self,
        target_freq: str,
        time_col: Optional[str] = None,
        agg_method: AggMethod = "mean",
        upsample_method: str = "linear",
        numeric_only: bool = True,
    ):
        self.target_freq = target_freq
        self.time_col = time_col
        self.agg_method = agg_method
        self.upsample_method = upsample_method
        self.numeric_only = numeric_only

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample df to target_freq. Returns new DataFrame with DatetimeIndex.
        """
        wdf = df.copy()

        # Set datetime index
        if self.time_col and self.time_col in wdf.columns:
            wdf[self.time_col] = pd.to_datetime(wdf[self.time_col])
            wdf = wdf.set_index(self.time_col)
        elif not isinstance(wdf.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have a DatetimeIndex or provide time_col."
            )

        if self.numeric_only:
            wdf = wdf.select_dtypes(include=[np.number])

        detected = self._detect_freq(wdf)
        detected_rank = _parse_freq_rank(detected) if detected else -1
        target_rank  = _parse_freq_rank(self.target_freq)

        if target_rank == -1 or detected_rank == -1 or target_rank >= detected_rank:
            # Downsampling (or same frequency)
            resampled = wdf.resample(self.target_freq).agg(self.agg_method)
        else:
            # Upsampling
            resampled = wdf.resample(self.target_freq).asfreq()
            if self.upsample_method == "linear":
                resampled = resampled.interpolate(method="linear")
            elif self.upsample_method == "pchip":
                resampled = resampled.interpolate(method="pchip")
            elif self.upsample_method == "ffill":
                resampled = resampled.ffill()
            elif self.upsample_method == "bfill":
                resampled = resampled.bfill()

        return resampled.dropna(how="all")

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_freq(df: pd.DataFrame) -> Optional[str]:
        """Infer the most common timedelta between consecutive index entries."""
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return None
        diffs = pd.Series(df.index).diff().dropna()
        mode_delta = diffs.mode().iloc[0]
        total_seconds = mode_delta.total_seconds()
        if total_seconds < 60:
            return f"{int(total_seconds)}s"
        if total_seconds < 3600:
            return f"{int(total_seconds//60)}min"
        if total_seconds < 86400:
            return f"{int(total_seconds//3600)}h"
        if total_seconds < 86400 * 7:
            return f"{int(total_seconds//86400)}D"
        if total_seconds < 86400 * 32:
            return "W"
        return "ME"

    @classmethod
    def auto_resample(
        cls,
        df: pd.DataFrame,
        target_freq: str,
        time_col: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        return cls(target_freq, time_col=time_col, **kwargs).fit_transform(df)
