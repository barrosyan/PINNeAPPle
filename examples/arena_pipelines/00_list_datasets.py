"""
00_list_datasets.py — List and inspect all available PINNeAPPle datasets.

Run with:
    cd examples/arena_pipelines
    python 00_list_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinneapple_data.datasets import list_datasets, load_dataset, dataset_ids

# ── List all datasets ────────────────────────────────────────────────────────
list_datasets()

# ── Show by category ─────────────────────────────────────────────────────────
for cat in ("physics", "geometry", "timeseries"):
    print(f"\n── {cat.upper()} IDs ──")
    print(dataset_ids(cat))

# ── Quick load test ──────────────────────────────────────────────────────────
test_ids = ["burgers_1d", "heat_1d", "lorenz63", "spring_mass_1dof",
            "naca0012", "cylinder_2d"]

print("\n── Quick load test ──────────────────────────────────────────────────────")
for did in test_ids:
    try:
        data = load_dataset(did)
        keys = ", ".join(f"{k}:{v.shape if hasattr(v,'shape') else v}"
                         for k, v in data.items()
                         if k not in ("description", "source", "pde") and
                         hasattr(v, "__len__"))
        print(f"  ✓ {did:<30} {keys}")
    except Exception as e:
        print(f"  ✗ {did:<30} ERROR: {e}")

# ── Real-world datasets (require network or optional libs) ───────────────────
print("\n── Real-world dataset test (may need network) ───────────────────────────")
rw_ids = ["sunspots", "air_passengers", "co2", "nasa_giss_temp",
          "sklearn_california_housing", "sklearn_diabetes",
          "seaborn_penguins", "airfoil_noise"]

for did in rw_ids:
    try:
        data = load_dataset(did)
        n = len(data.get("X", data.get("signal", [])))
        src = data.get("source", "?")
        print(f"  ✓ {did:<35} n={n}  src={src[:50]}")
    except Exception as e:
        print(f"  ✗ {did:<35} {e}")

# ── Physics simulation / CFD / materials / geoscience datasets ───────────────
print("\n── Physics simulation datasets (synthetic / NASA API) ───────────────────")
sim_ids = [
    "nasa_exoplanet_archive",   # live NASA API (falls back to synthetic)
    "nist_fluid_properties",    # water thermodynamics (NIST-like)
    "cfd_cylinder_drag",        # 2D cylinder Cd/Cl vs Re
    "seismic_waveform",         # synthetic 12-trace seismic gather
    "heat_conduction_rod",      # 1D FDM heat equation
    "turbulent_channel_flow",   # DNS channel flow Re_tau=180
    "materials_fatigue",        # S-N curve Al 6061-T6
    "orbit_propagation",        # Keplerian ISS-like orbit
    "plasma_fusion",            # ITER-like tokamak discharge
    "reaction_diffusion",       # Gray-Scott 2D pattern formation
]

for did in sim_ids:
    try:
        data = load_dataset(did)
        X = data.get("X", data.get("signal", []))
        shape = X.shape if hasattr(X, "shape") else len(X)
        src = data.get("source", "?")
        print(f"  ✓ {did:<30} X={shape}  src={src[:40]}")
    except Exception as e:
        print(f"  ✗ {did:<30} {e}")

print("\nDone.")
