"""
10 EDA tools for time series analysis.

1.  plot_trend            — trend + optional STL/rolling decomposition
2.  track_setpoint        — SP tracking: actual vs setpoint with deviation bands
3.  rolling_statistics    — rolling mean, std, min, max overlay
4.  step_response         — step response analysis (FOPDT estimation)
5.  power_spectrum        — FFT power spectrum with dominant frequency annotation
6.  plot_acf_pacf         — ACF and PACF side by side
7.  stationarity_report   — ADF / KPSS / PP textual dashboard
8.  changepoint_plot      — change point detection visualization
9.  cross_correlation     — pairwise cross-correlation heatmap / lag plot
10. rga_matrix            — Relative Gain Array (MV-CV interaction matrix)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure


def _ax(axes, i):
    return axes[i] if hasattr(axes, "__len__") else axes


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trend plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_trend(
    y: Union[pd.Series, np.ndarray],
    *,
    window: int = 24,
    title: str = "Trend Analysis",
    figsize: Tuple = (12, 6),
    show: bool = False,
) -> Figure:
    """Raw series + rolling mean + rolling std band."""
    s = pd.Series(y).reset_index(drop=True)
    mu = s.rolling(window, center=True, min_periods=1).mean()
    sd = s.rolling(window, center=True, min_periods=1).std()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, color="#4c9be8", lw=0.8, alpha=0.7, label="Series")
    ax.plot(mu.index, mu.values, color="#f78166", lw=1.8, label=f"Rolling mean (w={window})")
    ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.18, color="#f78166", label="±1σ")
    ax.set_title(title); ax.set_xlabel("Index"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Setpoint tracking
# ─────────────────────────────────────────────────────────────────────────────

def track_setpoint(
    actual: Union[pd.Series, np.ndarray],
    setpoint: Union[pd.Series, np.ndarray, float],
    *,
    tolerance: float = 0.05,
    title: str = "Setpoint Tracking",
    figsize: Tuple = (12, 5),
    show: bool = False,
) -> Figure:
    """Actual vs setpoint with deviation shading and tolerance band."""
    a = np.asarray(actual).ravel()
    sp = np.full_like(a, float(setpoint)) if np.isscalar(setpoint) else np.asarray(setpoint).ravel()
    err = a - sp
    x = np.arange(len(a))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.plot(x, a, color="#4c9be8", lw=1.2, label="Actual")
    ax1.plot(x, sp, color="#f0a500", lw=1.5, ls="--", label="Setpoint")
    sp_abs = np.abs(sp)
    ax1.fill_between(x, sp - tolerance * sp_abs, sp + tolerance * sp_abs,
                     alpha=0.12, color="#3fb950", label=f"±{tolerance*100:.0f}% tolerance")
    ax1.set_title(title); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.2)

    ax2.fill_between(x, err, 0, where=(err >= 0), color="#f78166", alpha=0.6, label="Overshoot")
    ax2.fill_between(x, err, 0, where=(err < 0),  color="#4c9be8", alpha=0.6, label="Undershoot")
    ax2.axhline(0, color="white", lw=0.8)
    ax2.set_ylabel("Error"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rolling statistics
# ─────────────────────────────────────────────────────────────────────────────

def rolling_statistics(
    y: Union[pd.Series, np.ndarray],
    windows: Sequence[int] = (8, 24, 96),
    *,
    stats: Sequence[str] = ("mean", "std"),
    title: str = "Rolling Statistics",
    figsize: Tuple = (12, 6),
    show: bool = False,
) -> Figure:
    """Multi-window rolling statistics overlay."""
    s = pd.Series(y).reset_index(drop=True)
    n_stats = len(stats)
    fig, axes = plt.subplots(n_stats, 1, figsize=(figsize[0], figsize[1] * n_stats / 2), sharex=True)
    if n_stats == 1: axes = [axes]
    palette = plt.cm.Set1(np.linspace(0, 0.8, len(windows)))

    for ax, stat in zip(axes, stats):
        ax.plot(s.index, s.values, color="gray", lw=0.6, alpha=0.4, label="Raw")
        for w, c in zip(windows, palette):
            r = getattr(s.rolling(w, center=True, min_periods=1), stat)()
            ax.plot(r.index, r.values, lw=1.3, color=c, label=f"w={w}")
        ax.set_ylabel(stat.capitalize()); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    axes[0].set_title(title)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Step response
# ─────────────────────────────────────────────────────────────────────────────

def step_response(
    y: Union[pd.Series, np.ndarray],
    step_idx: Optional[int] = None,
    *,
    title: str = "Step Response",
    figsize: Tuple = (12, 5),
    show: bool = False,
) -> Figure:
    """
    Visualize system response to a detected or specified step.
    Estimates: steady-state gain, rise time (10%→90%), settling time (±2%).
    """
    s = np.asarray(y).ravel().astype(float)
    if step_idx is None:
        diff = np.abs(np.diff(s))
        step_idx = int(np.argmax(diff))

    pre  = s[:step_idx]
    post = s[step_idx:]
    y0   = np.median(pre[-max(1, len(pre)//5):])
    yf   = np.median(post[-max(1, len(post)//5):])
    gain = yf - y0

    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, s, color="#4c9be8", lw=1.2, label="Response")
    ax.axvline(step_idx, color="#f78166", ls="--", lw=1.2, label=f"Step @ {step_idx}")
    ax.axhline(y0, color="#8b949e", ls=":", lw=1)
    ax.axhline(yf, color="#3fb950", ls=":", lw=1, label=f"SS gain = {gain:.4g}")
    if abs(gain) > 1e-10:
        lo10 = y0 + 0.1 * gain; hi90 = y0 + 0.9 * gain
        idx10 = next((i for i, v in enumerate(s[step_idx:], step_idx) if
                      (gain > 0 and v >= lo10) or (gain < 0 and v <= lo10)), None)
        idx90 = next((i for i, v in enumerate(s[step_idx:], step_idx) if
                      (gain > 0 and v >= hi90) or (gain < 0 and v <= hi90)), None)
        if idx10 and idx90:
            ax.axvspan(idx10, idx90, alpha=0.1, color="#ffa657", label=f"Rise {idx90-idx10} samples")
    ax.set_title(title); ax.set_xlabel("Sample"); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. FFT power spectrum
# ─────────────────────────────────────────────────────────────────────────────

def power_spectrum(
    y: Union[pd.Series, np.ndarray],
    *,
    fs: float = 1.0,
    n_top: int = 5,
    title: str = "FFT Power Spectrum",
    figsize: Tuple = (12, 5),
    show: bool = False,
) -> Figure:
    """Log-scale FFT power spectrum with top-N dominant frequencies annotated."""
    s = np.asarray(y).ravel().astype(float)
    N = len(s)
    fft_vals = np.fft.rfft(s - s.mean())
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    power = (np.abs(fft_vals) ** 2) / N

    top_idx = np.argsort(power)[-n_top:][::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(freqs, power, color="#4c9be8", lw=1.0)
    for i in top_idx:
        ax.axvline(freqs[i], color="#f78166", lw=0.8, alpha=0.7)
        ax.annotate(f"{freqs[i]:.4g}", xy=(freqs[i], power[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7, color="#f78166")
    ax.set_xlabel(f"Frequency (fs={fs})"); ax.set_ylabel("Power")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACF / PACF
# ─────────────────────────────────────────────────────────────────────────────

def plot_acf_pacf(
    y: Union[pd.Series, np.ndarray],
    nlags: int = 40,
    *,
    alpha: float = 0.05,
    title: str = "ACF / PACF",
    figsize: Tuple = (12, 5),
    show: bool = False,
) -> Figure:
    """Side-by-side ACF and PACF with confidence bounds."""
    s = np.asarray(y).ravel().astype(float)
    n = len(s)
    conf = 1.96 / np.sqrt(n)

    def _acf_manual(x, nlags):
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        return np.array([np.dot(x[k:], x[:-k if k else len(x)]) / (len(x) * c0)
                         for k in range(nlags + 1)])

    acf_vals = _acf_manual(s, nlags)
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(s, nlags=nlags, method="ywmle")
    except Exception:
        pacf_vals = acf_vals.copy()  # fallback

    lags = np.arange(nlags + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)
    for ax, vals, name in [(ax1, acf_vals, "ACF"), (ax2, pacf_vals, "PACF")]:
        ax.vlines(lags, 0, vals, color="#4c9be8", lw=1.2)
        ax.scatter(lags, vals, color="#4c9be8", s=20, zorder=3)
        ax.axhline(conf, color="#f78166", ls="--", lw=0.8)
        ax.axhline(-conf, color="#f78166", ls="--", lw=0.8)
        ax.axhline(0, color="white", lw=0.6)
        ax.set_title(name); ax.set_xlabel("Lag"); ax.grid(True, alpha=0.2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Stationarity report
# ─────────────────────────────────────────────────────────────────────────────

def stationarity_report(
    y: Union[pd.Series, np.ndarray],
    *,
    title: str = "Stationarity Tests",
    figsize: Tuple = (10, 4),
    show: bool = False,
) -> Figure:
    """Runs ADF, KPSS, and PP tests and renders a summary table."""
    s = np.asarray(y).ravel().astype(float)
    rows = []
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        rows.append(("ADF", f"{adf_stat:.4f}", f"{adf_p:.4f}", "Stationary" if adf_p < 0.05 else "Non-stationary"))
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
        rows.append(("KPSS", f"{kpss_stat:.4f}", f"{kpss_p:.4f}", "Stationary" if kpss_p > 0.05 else "Non-stationary"))
    except Exception as e:
        rows.append(("Error", str(e), "", ""))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Test", "Statistic", "p-value", "Verdict"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1c2330"); cell.set_text_props(color="#58a6ff", weight="bold")
        elif r % 2:
            cell.set_facecolor("#161b22")
        else:
            cell.set_facecolor("#0d1117")
        cell.set_text_props(color="#e6edf3")
        cell.set_edgecolor("#30363d")
    ax.set_title(title, pad=20)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Change point detection
# ─────────────────────────────────────────────────────────────────────────────

def changepoint_plot(
    y: Union[pd.Series, np.ndarray],
    n_bkps: int = 3,
    *,
    model: str = "rbf",
    title: str = "Change Point Detection",
    figsize: Tuple = (12, 5),
    show: bool = False,
) -> Figure:
    """Visualize change points detected via the ruptures library."""
    s = np.asarray(y).ravel().astype(float)
    breakpoints = []
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model=model).fit(s)
        breakpoints = algo.predict(n_bkps=n_bkps)[:-1]
    except ImportError:
        # Fallback: simple variance-based segmentation
        step = max(1, len(s) // (n_bkps + 1))
        breakpoints = list(range(step, len(s) - step, step))[:n_bkps]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s, color="#4c9be8", lw=1.0, alpha=0.8, label="Series")
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(breakpoints) + 1))
    prev = 0
    for i, bp in enumerate(breakpoints + [len(s)]):
        ax.axvspan(prev, bp, alpha=0.08, color=colors[i])
        if bp < len(s):
            ax.axvline(bp, color=colors[i], lw=1.5, ls="--", label=f"CP @ {bp}")
        prev = bp
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. Cross-correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def cross_correlation(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    max_lag: int = 20,
    *,
    title: str = "Cross-Correlation",
    figsize: Tuple = (12, 6),
    show: bool = False,
) -> Figure:
    """
    Heatmap of peak cross-correlation values and lag at peak between all column pairs.
    """
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(cols)
    corr_matrix = np.zeros((n, n))
    lag_matrix  = np.zeros((n, n), dtype=int)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            xi = df[ci].dropna().values.astype(float)
            xj = df[cj].dropna().values.astype(float)
            min_len = min(len(xi), len(xj))
            xi, xj = xi[:min_len], xj[:min_len]
            xi = (xi - xi.mean()) / (xi.std() + 1e-10)
            xj = (xj - xj.mean()) / (xj.std() + 1e-10)
            ccf = np.correlate(xi, xj, mode="full") / min_len
            center = len(ccf) // 2
            window = ccf[center - max_lag: center + max_lag + 1]
            best = np.argmax(np.abs(window))
            corr_matrix[i, j] = window[best]
            lag_matrix[i, j]  = best - max_lag

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for ax, mat, label, fmt in [
        (ax1, corr_matrix, "Peak Correlation", ".2f"),
        (ax2, lag_matrix,  "Lag at Peak",     "d"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r" if "Corr" in label else "coolwarm",
                       vmin=-1 if "Corr" in label else None,
                       vmax= 1 if "Corr" in label else None)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(cols, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:{fmt}}", ha="center", va="center", fontsize=7)
        ax.set_title(label)
    fig.suptitle(title); fig.tight_layout()
    if show: plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. RGA matrix (Relative Gain Array)
# ─────────────────────────────────────────────────────────────────────────────

def rga_matrix(
    K: np.ndarray,
    mv_labels: Optional[List[str]] = None,
    cv_labels: Optional[List[str]] = None,
    *,
    title: str = "Relative Gain Array (RGA)",
    figsize: Tuple = (8, 6),
    show: bool = False,
) -> Figure:
    """
    Compute and visualize the Bristol Relative Gain Array.

    RGA = K ⊙ (K⁻¹)ᵀ  where ⊙ is element-wise multiplication.

    Parameters
    ----------
    K          : (n_cv, n_mv) steady-state gain matrix
    mv_labels  : manipulated variable names
    cv_labels  : controlled variable names
    """
    K = np.atleast_2d(K).astype(float)
    n_cv, n_mv = K.shape
    if K.shape[0] == K.shape[1]:
        Kinv = np.linalg.inv(K)
    else:
        Kinv = np.linalg.pinv(K)
    rga = K * Kinv.T

    mv_labels = mv_labels or [f"MV{i+1}" for i in range(n_mv)]
    cv_labels = cv_labels or [f"CV{i+1}" for i in range(n_cv)]

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(2.0, np.abs(rga).max())
    im = ax.imshow(rga, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="RGA element λᵢⱼ")
    ax.set_xticks(range(n_mv)); ax.set_xticklabels(mv_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_cv)); ax.set_yticklabels(cv_labels)
    for i in range(n_cv):
        for j in range(n_mv):
            color = "white" if abs(rga[i, j]) > 0.5 * vmax else "black"
            ax.text(j, i, f"{rga[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_title(title)
    ax.set_xlabel("Manipulated Variables"); ax.set_ylabel("Controlled Variables")

    # Annotation: ideal pairing (λ closest to 1)
    pairs = []
    used_mv = set()
    rga_copy = rga.copy()
    for _ in range(min(n_cv, n_mv)):
        idx = np.unravel_index(np.argmin(np.abs(rga_copy - 1)), rga_copy.shape)
        pairs.append(idx)
        rga_copy[idx[0], :] = np.inf
        rga_copy[:, idx[1]] = np.inf
    for (i, j) in pairs:
        rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                              fill=False, edgecolor="#3fb950", linewidth=2.5)
        ax.add_patch(rect)

    fig.tight_layout()
    if show: plt.show()
    return fig
