"""
02_timeseries_benchmark.py — TimeSeriesBenchmarkSpec examples

Demonstrates:
  1. Synthetic dataset — Lorenz63 attractor (LSTM vs TCN vs NBeats)
  2. Real dataset — Sunspots (LSTM vs MLP)
  3. Real dataset — AirPassengers (LSTM only)
  4. Multi-variate — spring_mass_2dof (GRU)

Run with:
    cd examples/arena_pipelines
    python 02_timeseries_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneaple_tools.benchmark_suite import TimeSeriesBenchmarkSpec

OUT = Path("outputs") / "timeseries"

# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — Lorenz63 chaotic attractor: LSTM vs TCN vs NBeats
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 1 — Lorenz63 Attractor (LSTM vs TCN vs NBeats)")
print("="*60)

spec1 = TimeSeriesBenchmarkSpec(
    source    = "lorenz63",
    models    = ["lstm", "tcn", "nbeats"],
    metrics   = ["mse", "mae", "rmse"],
    horizon   = 20,
    lookback  = 50,
    test_size = 0.2,
    plots     = True,
    epochs    = 60,
    lr        = 1e-3,
    seed      = 42,
    output_dir = str(OUT),
)

report1 = spec1.run()
report1.save(OUT / "lorenz63_report.json")
print(f"\n  Best model: {report1.best_model}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — Sunspots (real data, LSTM vs MLP)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 2 — Sunspots (real, LSTM vs MLP)")
print("="*60)

spec2 = TimeSeriesBenchmarkSpec(
    source    = "sunspots",
    models    = ["lstm", "transformer"],
    metrics   = ["mse", "rmse", "mae"],
    horizon   = 12,         # 12 months ahead
    lookback  = 60,         # 5-year window
    test_size = 0.2,
    plots     = True,
    epochs    = 80,
    lr        = 1e-3,
    seed      = 0,
    output_dir = str(OUT),
)

report2 = spec2.run()
report2.save(OUT / "sunspots_report.json")
print(f"\n  Best model: {report2.best_model}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — Airline Passengers (classical Box-Jenkins, LSTM vs NBeats)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 3 — AirPassengers (LSTM vs NBeats)")
print("="*60)

spec3 = TimeSeriesBenchmarkSpec(
    source    = "air_passengers",
    models    = ["lstm", "nbeats"],
    metrics   = ["mse", "rmse", "mape"],
    horizon   = 12,         # 12 months ahead
    lookback  = 24,
    test_size = 0.25,
    plots     = True,
    epochs    = 100,
    lr        = 5e-4,
    seed      = 1,
    output_dir = str(OUT),
)

report3 = spec3.run()
report3.save(OUT / "air_passengers_report.json")


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — Spring-mass 2DOF (multi-variate, GRU)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  Example 4 — Spring-Mass 2DOF (multi-variate GRU vs Transformer)")
print("="*60)

spec4 = TimeSeriesBenchmarkSpec(
    source      = "spring_mass_2dof",
    models      = ["gru", "transformer"],
    metrics     = ["mse", "mae"],
    target_cols = None,     # all 4 states (x1, v1, x2, v2)
    horizon     = 20,
    lookback    = 40,
    test_size   = 0.2,
    plots       = True,
    epochs      = 80,
    lr          = 1e-3,
    seed        = 42,
    output_dir  = str(OUT),
)

report4 = spec4.run()
report4.save(OUT / "spring_mass_report.json")

print("\n" + "="*60)
print("  All time-series benchmarks complete.")
print(f"  Results saved to: {OUT.resolve()}")
print("="*60)
