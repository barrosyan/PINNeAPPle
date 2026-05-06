"""Audit + feature engineering demo (no model training).

Shows:
  - building an AuditReport (stationarity, autocorrelation, heteroscedasticity, regime changes)
  - generating lag/rolling/Fourier features with TSFeatureEngineer
  - keeping things safe if optional deps aren't installed (statsmodels/ruptures)

Run:
  python examples/pinneaple_timeseries/03_audit_and_features.py
"""

from __future__ import annotations

import math
import numpy as np

from pinneaple_timeseries import TSAuditor, TSFeatureEngineer, ForecastProblemSpec


def make_series(T: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    # piecewise regime + seasonal
    y = 0.6 * np.sin(2 * math.pi * t / 24.0) + 0.15 * rng.standard_normal(T)
    y[t > 1100] += 0.7  # mean shift regime break
    y[t > 1500] *= 1.4  # variance shift
    return y.astype(float)


def main() -> None:
    y = make_series()
    t = np.arange(len(y), dtype=float)

    problem = ForecastProblemSpec(
        name="audit_demo",
        target_name="y",
        freq="H",  # hourly (used only as metadata)
        horizon=24,
        input_len=96,
    )

    auditor = TSAuditor()
    report = auditor.run(y, meta={"name": problem.name, "T": len(y), "freq": problem.freq})

    print("=== Audit summary ===")
    for section in report.sections:
        print(f"\n[{section.name}]")

        # Each section contains a list of entries. Keep printing lightweight.
        for entry in section.items[:5]:
            status = entry.get("status", "")
            key = entry.get("key", "")
            msg = entry.get("message", "")
            print(f"- {key:18s} {status:10s} {msg}")

        if len(section.items) > 5:
            print(f"  ... (+{len(section.items) - 5} more)")

    print("\n=== Feature engineering ===")
    fe = TSFeatureEngineer()
    feats = fe.transform(y=y, t=t)

    # Show a compact preview
    for k in sorted(feats.keys())[:10]:
        arr = feats[k]
        nan_pct = float(np.mean(np.isnan(arr))) * 100.0
        print(f"- {k:16s} shape={arr.shape}  NaNs={nan_pct:5.1f}%")

    print("\nTip: feed these features to your model (exogenous regressors) or to a downstream scaler.")


if __name__ == "__main__":
    main()