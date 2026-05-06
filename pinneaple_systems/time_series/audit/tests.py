from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .report import AuditReport


def _as_1d(y) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D series, got shape {arr.shape}")
    return arr


def _clean_series(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Clean a 1D series to make statsmodels tests robust.

    Strategy:
      - convert inf -> nan
      - drop nan values

    Returns:
      x_clean, info dict with counts.
    """
    x = np.asarray(y, dtype=float).reshape(-1)
    n_total = int(len(x))
    inf_mask = ~np.isfinite(x)  # True for nan or inf
    n_bad = int(inf_mask.sum())

    if n_bad > 0:
        x = x.copy()
        x[inf_mask] = np.nan

    n_nan = int(np.isnan(x).sum())
    x_clean = x[~np.isnan(x)]
    info = {
        "n_total": n_total,
        "n_bad_nonfinite": n_bad,
        "n_nan": n_nan,
        "n_clean": int(len(x_clean)),
    }
    return x_clean, info


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def _require_len(x: np.ndarray, n_min: int, name: str) -> Optional[Dict[str, Any]]:
    if len(x) < int(n_min):
        return {"ok": False, "error": f"{name} requires at least {n_min} clean points, got {len(x)}"}
    return None


def adf_test(y, *, autolag: str = "AIC", min_n: int = 20) -> Dict[str, Any]:
    if not _has_statsmodels():
        return {"ok": False, "error": "statsmodels is not installed"}

    from statsmodels.tsa.stattools import adfuller

    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "ADF")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        stat, pval, usedlag, nobs, crit, icbest = adfuller(x, autolag=autolag)
        return {
            "ok": True,
            "stat": float(stat),
            "pvalue": float(pval),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()},
            "icbest": float(icbest),
            "cleaning": info,
        }
    except Exception as e:
        return {"ok": False, "error": f"ADF failed: {type(e).__name__}: {e}", "cleaning": info}


def kpss_test(y, *, regression: str = "c", nlags: str = "auto", min_n: int = 20) -> Dict[str, Any]:
    if not _has_statsmodels():
        return {"ok": False, "error": "statsmodels is not installed"}

    from statsmodels.tsa.stattools import kpss

    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "KPSS")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        stat, pval, lags, crit = kpss(x, regression=regression, nlags=nlags)
        return {
            "ok": True,
            "stat": float(stat),
            "pvalue": float(pval),
            "lags": int(lags),
            "crit": {k: float(v) for k, v in crit.items()},
            "cleaning": info,
        }
    except Exception as e:
        return {"ok": False, "error": f"KPSS failed: {type(e).__name__}: {e}", "cleaning": info}


def pp_test(y, min_n: int = 20) -> Dict[str, Any]:
    """
    Phillips–Perron test.

    Availability depends on the statsmodels version. If unavailable, we return a clear error.
    """
    if not _has_statsmodels():
        return {"ok": False, "error": "statsmodels is not installed"}

    try:
        from statsmodels.tsa.stattools import phillips_perron  # type: ignore
    except Exception:
        return {"ok": False, "error": "Phillips–Perron is not available in this statsmodels version"}

    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "PP")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        res = phillips_perron(x)
        out = {"ok": True, "cleaning": info}
        for k in ["stat", "pvalue"]:
            if hasattr(res, k):
                out[k] = float(getattr(res, k))
        return out
    except Exception as e:
        return {"ok": False, "error": f"PP failed: {type(e).__name__}: {e}", "cleaning": info}


def arch_test(y, *, lags: int = 12, min_n: int = 30) -> Dict[str, Any]:
    if not _has_statsmodels():
        return {"ok": False, "error": "statsmodels is not installed"}

    from statsmodels.stats.diagnostic import het_arch

    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "ARCH")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        stat, pval, fstat, fpval = het_arch(x, nlags=int(lags))
        return {
            "ok": True,
            "lm_stat": float(stat),
            "lm_pvalue": float(pval),
            "f_stat": float(fstat),
            "f_pvalue": float(fpval),
            "cleaning": info,
        }
    except Exception as e:
        return {"ok": False, "error": f"ARCH failed: {type(e).__name__}: {e}", "cleaning": info}


def acf_pacf(y, *, nlags: int = 40, min_n: int = 50) -> Dict[str, Any]:
    if not _has_statsmodels():
        return {"ok": False, "error": "statsmodels is not installed"}

    from statsmodels.tsa.stattools import acf, pacf

    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, max(min_n, nlags + 5), "ACF/PACF")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        a = acf(x, nlags=int(nlags), fft=True)
        p = pacf(x, nlags=int(nlags), method="yw")
        return {"ok": True, "acf": [float(v) for v in a], "pacf": [float(v) for v in p], "cleaning": info}
    except Exception as e:
        return {"ok": False, "error": f"ACF/PACF failed: {type(e).__name__}: {e}", "cleaning": info}


def chow_test_mean_shift(y, break_idx: int, min_n: int = 20) -> Dict[str, Any]:
    """
    Simple Chow test for a mean shift using an intercept-only model.

    Note:
      We clean NaN/inf by dropping them. This changes the effective indexing.
      For strict "time index" breakpoints, run this test on an imputed series instead.
    """
    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "Chow")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    n = len(x)
    b = int(break_idx)
    # If user gave break_idx based on original length, we map it proportionally.
    if b >= n:
        b = int((n - 1) * 0.5)

    if b <= 2 or b >= n - 2:
        return {"ok": False, "error": "break_idx is too close to the series ends after cleaning", "cleaning": info}

    y1, y2 = x[:b], x[b:]
    c_all = x.mean()
    c1, c2 = y1.mean(), y2.mean()

    rss_all = float(((x - c_all) ** 2).sum())
    rss1 = float(((y1 - c1) ** 2).sum())
    rss2 = float(((y2 - c2) ** 2).sum())

    k = 1
    num = (rss_all - (rss1 + rss2)) / k
    den = (rss1 + rss2) / (n - 2 * k)
    if den <= 0:
        return {"ok": False, "error": "degenerate denominator", "cleaning": info}
    F = float(num / den)

    pval = None
    try:
        from scipy.stats import f as fdist
        pval = float(1.0 - fdist.cdf(F, k, n - 2 * k))
    except Exception:
        pass

    return {
        "ok": True,
        "F": F,
        "pvalue": pval,
        "break_idx": int(b),
        "rss_all": rss_all,
        "rss1": rss1,
        "rss2": rss2,
        "cleaning": info,
    }


def changepoints_ruptures(y, *, model: str = "l2", max_breaks: int = 5, min_n: int = 50) -> Dict[str, Any]:
    """
    Multi-break approximation via `ruptures` (if installed).
    """
    raw = _as_1d(y)
    x, info = _clean_series(raw)

    too_short = _require_len(x, min_n, "Changepoints")
    if too_short is not None:
        too_short["cleaning"] = info
        return too_short

    try:
        import ruptures as rpt
    except Exception:
        return {"ok": False, "error": "ruptures is not installed", "cleaning": info}

    try:
        algo = rpt.Pelt(model=model).fit(x)
        penalty = 3.0 * np.log(len(x))
        bkps = algo.predict(pen=penalty)
        breaks = [int(b) for b in bkps[:-1]]
        if len(breaks) > int(max_breaks):
            breaks = breaks[: int(max_breaks)]
        return {"ok": True, "breaks": breaks, "penalty": float(penalty), "model": model, "cleaning": info}
    except Exception as e:
        return {"ok": False, "error": f"ruptures failed: {type(e).__name__}: {e}", "cleaning": info}


@dataclass
class TSAuditor:
    nlags: int = 40
    arch_lags: int = 12

    def run(self, y, *, meta: Optional[Dict[str, Any]] = None) -> AuditReport:
        report = AuditReport(meta=meta or {})

        report.add("stationarity", "adf", adf_test(y))
        report.add("stationarity", "kpss", kpss_test(y))
        report.add("stationarity", "pp", pp_test(y))

        report.add("heteroscedasticity", "arch", arch_test(y, lags=self.arch_lags))
        report.add("autocorrelation", "acf_pacf", acf_pacf(y, nlags=self.nlags))

        raw = _as_1d(y)
        # midpoint based on original length; chow() will map if needed after cleaning
        mid = len(raw) // 2
        report.add("regime_breaks", "chow_mid", chow_test_mean_shift(y, break_idx=mid))
        report.add("regime_breaks", "changepoints", changepoints_ruptures(y))

        return report