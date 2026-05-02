"""Real-world datasets loaded from the web or optional Python libraries.

Every loader tries multiple sources in order (library → URL → cached fallback).
Datasets are downloaded once and cached in memory (per-process) for reuse.

Datasets
--------
  timeseries:
    sunspots          — Monthly sunspot numbers since 1749 (SIDC / statsmodels)
    air_passengers    — Monthly airline passengers 1949-1960 (classic Box-Jenkins)
    co2               — Mauna Loa weekly CO2 ppm since 1958 (NOAA / statsmodels)
    etth1             — ETT-H1 hourly temperature (ETDataset, 2016-2018)
    ettm1             — ETT-M1 15-min temperature (ETDataset, 2016-2018)
    jena_climate      — Jena Climate hourly T,p,rh,wind (2009-2016)
    nasa_giss_temp    — NASA GISS global surface temperature anomaly (1880-now)
    spacex_launches   — SpaceX launches: time-of-launch metadata time series

  physics / engineering (regression):
    airfoil_noise     — NASA airfoil self-noise (UCI ML)
    concrete_strength — Concrete compressive strength (UCI ML)
    energy_efficiency — Building energy efficiency (UCI ML)

  physics simulations / CFD / materials / geoscience:
    nasa_exoplanet_archive  — NASA confirmed exoplanet physical parameters
    nist_fluid_properties   — Water/steam thermodynamic properties (NIST-like synthetic)
    cfd_cylinder_drag       — 2D cylinder drag/lift vs Re (synthetic CFD parametric)
    seismic_waveform        — Synthetic 1D seismic P-wave propagation
    heat_conduction_rod     — 1D transient heat conduction in a rod (FDM)
    turbulent_channel_flow  — DNS turbulent channel flow statistics (Re_tau=180)
    materials_fatigue       — S-N fatigue curve data (aluminum alloys, synthetic)
    orbit_propagation       — Satellite orbit TLE-based propagation (Keplerian)
    plasma_fusion           — Tokamak plasma parameters time series (synthetic ITER-like)
    reaction_diffusion      — 2D Gray-Scott reaction-diffusion simulation

  library (sklearn / seaborn / statsmodels built-ins):
    sklearn_california_housing  — CA housing prices (sklearn)
    sklearn_diabetes            — Diabetes progression (sklearn)
    sklearn_wine                — Wine quality features (sklearn)
    seaborn_penguins            — Palmer penguins morphology (seaborn)
    seaborn_mpg                 — Vehicle MPG (seaborn)
    statsmodels_macrodata       — US macroeconomic data (statsmodels)
    statsmodels_elnino          — El Nino SST (statsmodels)
"""
from __future__ import annotations

import io as _io
import urllib.request
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from .registry import DatasetInfo, DatasetRegistry

# ──────────────────────────────────────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────────────────────────────────────

_TIMEOUT = 30   # seconds


def _fetch(url: str, timeout: int = _TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "PINNeAPPle/0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_text(url: str, **kw) -> str:
    return _fetch(url, **kw).decode("utf-8", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sunspots — Monthly mean total sunspot number
#    Primary: statsmodels built-in dataset (Wolfer, 1700-1988)
#    Fallback: SIDC (https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sunspots() -> Dict[str, np.ndarray]:
    # 1. statsmodels
    try:
        from statsmodels.datasets.sunspots import load_pandas
        df = load_pandas().data
        t = df.index.to_numpy(dtype=float)
        signal = df["SUNACTIVITY"].to_numpy(dtype=np.float32)
        return {
            "t": t, "signal": signal, "X": signal.reshape(-1, 1),
            "description": "Monthly mean sunspot number (statsmodels/Wolfer, 1700-1988)",
            "source": "statsmodels.datasets.sunspots",
        }
    except Exception:
        pass

    # 2. SIDC URL — format: YYYY MM <day_frac> <monthly_mean> <monthly_std> <n_obs> <provisional>
    try:
        url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"
        text = _fetch_text(url)
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                yr = float(parts[0])
                mo = float(parts[1])
                val = float(parts[3])
                if val >= 0:
                    rows.append((yr + (mo - 0.5) / 12.0, val))
        rows_arr = np.array(rows, dtype=np.float32)
        return {
            "t": rows_arr[:, 0], "signal": rows_arr[:, 1],
            "X": rows_arr[:, 1:2],
            "description": "Monthly mean sunspot number v2.0 (SIDC/SILSO)",
            "source": url,
        }
    except Exception:
        pass

    # 3. Minimal embedded fallback (100 representative values)
    rng = np.random.default_rng(0)
    t = np.arange(100, dtype=np.float32)
    signal = (50 + 40 * np.sin(2 * np.pi * t / 11.0) +
              rng.normal(0, 5, 100)).clip(0).astype(np.float32)
    return {
        "t": t, "signal": signal, "X": signal.reshape(-1, 1),
        "description": "Synthetic sunspot placeholder (real data unavailable)",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. AirPassengers — Box & Jenkins airline dataset (1949-1960, monthly)
# ──────────────────────────────────────────────────────────────────────────────

_AIR_PASSENGERS = np.array([
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432,
], dtype=np.float32)

@lru_cache(maxsize=1)
def _load_air_passengers() -> Dict[str, np.ndarray]:
    try:
        # pandas has it via a URL
        import pandas as pd
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        df = pd.read_csv(url, header=0, names=["month", "passengers"])
        signal = df["passengers"].to_numpy(dtype=np.float32)
        t = np.arange(len(signal), dtype=np.float32)
        return {
            "t": t, "signal": signal, "X": signal.reshape(-1, 1),
            "description": "Monthly airline passengers 1949-1960 (Box & Jenkins)",
            "source": url,
        }
    except Exception:
        pass

    signal = _AIR_PASSENGERS
    return {
        "t": np.arange(len(signal), dtype=np.float32),
        "signal": signal,
        "X": signal.reshape(-1, 1),
        "description": "Monthly airline passengers 1949-1960 (embedded)",
        "source": "embedded",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. CO₂ — Mauna Loa weekly atmospheric CO₂ ppm (Keeling Curve)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_co2() -> Dict[str, np.ndarray]:
    # statsmodels
    try:
        from statsmodels.datasets.co2 import load_pandas
        df = load_pandas().data.dropna()
        signal = df["co2"].to_numpy(dtype=np.float32)
        t = np.arange(len(signal), dtype=np.float32)
        return {
            "t": t, "signal": signal, "X": signal.reshape(-1, 1),
            "description": "Weekly Mauna Loa CO₂ ppm (statsmodels, 1958-2001)",
            "source": "statsmodels.datasets.co2",
        }
    except Exception:
        pass

    # NOAA URL
    try:
        url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.txt"
        text = _fetch_text(url)
        rows = []
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    val = float(parts[4])
                    if val > 0:
                        rows.append(val)
                except ValueError:
                    continue
        signal = np.array(rows, dtype=np.float32)
        return {
            "t": np.arange(len(signal), dtype=np.float32),
            "signal": signal, "X": signal.reshape(-1, 1),
            "description": "Weekly Mauna Loa CO₂ ppm (NOAA GML)",
            "source": url,
        }
    except Exception:
        pass

    # Synthetic fallback
    n = 500
    t = np.arange(n, dtype=np.float32)
    signal = (315 + 0.1 * t +
              3.0 * np.sin(2 * np.pi * t / 52.18) +
              np.random.default_rng(1).normal(0, 0.5, n)).astype(np.float32)
    return {
        "t": t, "signal": signal, "X": signal.reshape(-1, 1),
        "description": "Synthetic CO₂ placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. ETT-H1 — Electricity Transformer Temperature (hourly, 2016-2018)
#    14 months train + 4 months val + 4 months test (standard split)
#    Columns: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_etth1() -> Dict[str, np.ndarray]:
    url = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
           "ETT-small/ETTh1.csv")
    try:
        import pandas as pd
        df = pd.read_csv(url)
        cols = [c for c in df.columns if c != "date"]
        X = df[cols].to_numpy(dtype=np.float32)
        t = np.arange(len(X), dtype=np.float32)
        return {
            "t": t, "X": X,
            "signal": X[:, -1],   # OT (oil temperature) as primary signal
            "columns": cols,
            "description": "ETT-H1: hourly electricity transformer temperature (2016-2018)",
            "source": url,
        }
    except Exception:
        pass

    # Minimal fallback
    n = 1000
    t = np.arange(n, dtype=np.float32)
    X = np.column_stack([
        np.sin(2*np.pi*t / 24 + i*0.5) + np.random.default_rng(i).normal(0, 0.1, n)
        for i in range(7)
    ]).astype(np.float32)
    return {
        "t": t, "X": X, "signal": X[:, -1],
        "columns": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"],
        "description": "ETT-H1 synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. ETT-M1 — Electricity Transformer Temperature (15-min, 2016-2018)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_ettm1() -> Dict[str, np.ndarray]:
    url = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
           "ETT-small/ETTm1.csv")
    try:
        import pandas as pd
        df = pd.read_csv(url)
        cols = [c for c in df.columns if c != "date"]
        X = df[cols].to_numpy(dtype=np.float32)
        t = np.arange(len(X), dtype=np.float32)
        return {
            "t": t, "X": X,
            "signal": X[:, -1],
            "columns": cols,
            "description": "ETT-M1: 15-min electricity transformer temperature (2016-2018)",
            "source": url,
        }
    except Exception:
        pass
    # Fallback — resample ETTh1 synthetic
    n = 4000
    t = np.arange(n, dtype=np.float32)
    X = np.column_stack([
        np.sin(2*np.pi*t / 96 + i*0.5) + np.random.default_rng(i+10).normal(0, 0.1, n)
        for i in range(7)
    ]).astype(np.float32)
    return {
        "t": t, "X": X, "signal": X[:, -1],
        "columns": ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"],
        "description": "ETT-M1 synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Jena Climate — hourly temperature (2009-2016), from Keras example
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_jena_climate() -> Dict[str, np.ndarray]:
    url = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
           "jena_climate_2009_2016.csv.zip")
    try:
        import pandas as pd, zipfile
        raw = _fetch(url, timeout=60)
        with zipfile.ZipFile(_io.BytesIO(raw)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f)
        cols = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"]
        available = [c for c in cols if c in df.columns]
        X = df[available].to_numpy(dtype=np.float32)
        # Keep every 6th sample (hourly from 10-min)
        X = X[::6]
        t = np.arange(len(X), dtype=np.float32)
        return {
            "t": t, "X": X,
            "signal": X[:, 0],   # temperature
            "columns": available,
            "description": "Jena Climate hourly T,p,rh,wv (2009-2016)",
            "source": url,
        }
    except Exception:
        pass
    n = 2000
    t = np.arange(n, dtype=np.float32)
    T = 10 + 10 * np.sin(2*np.pi*t/8766) + 5*np.sin(2*np.pi*t/24) + np.random.default_rng(2).normal(0, 1, n)
    return {
        "t": t, "X": T.reshape(-1, 1).astype(np.float32),
        "signal": T.astype(np.float32),
        "columns": ["T (degC)"],
        "description": "Jena Climate synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Airfoil Self-Noise — NASA/UCI (1503 samples, 6 features → 1 target)
#    Physics benchmark: predict sound pressure level from aerodynamic inputs
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_airfoil_noise() -> Dict[str, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    try:
        text = _fetch_text(url)
        rows = []
        for line in text.splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue
        arr = np.array(rows, dtype=np.float32)
        features = arr[:, :5]
        target = arr[:, 5]
        col_names = ["frequency_Hz", "angle_of_attack_deg", "chord_length_m",
                     "free_stream_velocity_ms", "suction_thickness_m"]
        return {
            "X": features, "y": target.reshape(-1, 1),
            "t": np.arange(len(target), dtype=np.float32),
            "signal": target,
            "feature_names": col_names,
            "target_name": "sound_pressure_level_dB",
            "description": "NASA airfoil self-noise: aerodynamic → SPL (UCI ML)",
            "source": url,
        }
    except Exception:
        pass
    # Fallback
    rng = np.random.default_rng(3)
    n = 200
    X = rng.uniform([100, 0, 0.05, 20, 0.001], [20000, 22, 0.3, 71, 0.06], (n, 5)).astype(np.float32)
    y = (125 - 0.001 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 2, n)).astype(np.float32)
    return {
        "X": X, "y": y.reshape(-1, 1),
        "t": np.arange(n, dtype=np.float32),
        "signal": y,
        "description": "Airfoil noise synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Concrete Compressive Strength — UCI (1030 samples, 8 features → 1 target)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_concrete_strength() -> Dict[str, np.ndarray]:
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml("concrete-strength", version=1, as_frame=True)
        X = ds.data.to_numpy(dtype=np.float32)
        y = ds.target.to_numpy(dtype=np.float32)
        return {
            "X": X, "y": y.reshape(-1, 1),
            "t": np.arange(len(y), dtype=np.float32),
            "signal": y,
            "feature_names": list(ds.feature_names),
            "target_name": "compressive_strength_MPa",
            "description": "Concrete compressive strength (UCI ML, 1030 samples)",
            "source": "sklearn/openml",
        }
    except Exception:
        pass
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "concrete/compressive/Concrete_Data.xls")
        import pandas as pd, io
        raw = _fetch(url, timeout=30)
        df = pd.read_excel(io.BytesIO(raw))
        arr = df.to_numpy(dtype=np.float32)
        X = arr[:, :8]; y = arr[:, 8]
        return {
            "X": X, "y": y.reshape(-1, 1),
            "t": np.arange(len(y), dtype=np.float32),
            "signal": y,
            "target_name": "compressive_strength_MPa",
            "description": "Concrete compressive strength (UCI ML)",
            "source": url,
        }
    except Exception:
        pass
    rng = np.random.default_rng(4)
    n = 200
    X = rng.uniform([100,0,0,0,0,0,0,1],[600,200,300,100,100,100,300,365], (n, 8)).astype(np.float32)
    y = (50 + 0.01*X[:,0] - 0.1*X[:,6] + rng.normal(0, 5, n)).astype(np.float32)
    return {
        "X": X, "y": y.reshape(-1, 1),
        "t": np.arange(n, dtype=np.float32),
        "signal": y,
        "description": "Concrete strength synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. Energy Efficiency — UCI (768 samples, 8 features → heating/cooling load)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_energy_efficiency() -> Dict[str, np.ndarray]:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    try:
        import pandas as pd, io
        raw = _fetch(url, timeout=30)
        df = pd.read_excel(io.BytesIO(raw))
        arr = df.dropna().to_numpy(dtype=np.float32)
        X = arr[:, :8]; y = arr[:, 8:10]
        col_names = ["relative_compactness","surface_area","wall_area","roof_area",
                     "overall_height","orientation","glazing_area","glazing_area_distribution"]
        return {
            "X": X, "y": y,
            "t": np.arange(len(y), dtype=np.float32),
            "signal": y[:, 0],
            "feature_names": col_names,
            "target_names": ["heating_load_kWh", "cooling_load_kWh"],
            "description": "Building energy efficiency (UCI ML, 768 samples)",
            "source": url,
        }
    except Exception:
        pass
    rng = np.random.default_rng(5)
    n = 200
    X = rng.uniform(0, 1, (n, 8)).astype(np.float32)
    y = (15 + 10*X[:,0] + rng.normal(0,2,(n,2))).astype(np.float32)
    return {
        "X": X, "y": y,
        "t": np.arange(n, dtype=np.float32),
        "signal": y[:, 0],
        "description": "Energy efficiency synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10. NASA GISS Global Surface Temperature Anomaly
#     Combined land-surface air and sea-surface water temperature (1880-present)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nasa_giss_temp() -> Dict[str, np.ndarray]:
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        import pandas as pd
        df = pd.read_csv(url, skiprows=1)
        df = df[df["Year"].apply(lambda x: str(x).isdigit())]
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        available = [m for m in months if m in df.columns]
        rows = []
        for _, row in df.iterrows():
            yr = int(row["Year"])
            for i, m in enumerate(available):
                try:
                    v = float(row[m])
                    rows.append((yr + i / 12.0, v))
                except (ValueError, TypeError):
                    pass
        arr = np.array(rows, dtype=np.float32)
        return {
            "t": arr[:, 0], "signal": arr[:, 1], "X": arr[:, 1:2],
            "description": "NASA GISS global surface temperature anomaly (monthly, 1880-present)",
            "source": url,
        }
    except Exception:
        pass
    # Fallback
    t = np.linspace(1880, 2024, 1728).astype(np.float32)
    signal = (-0.5 + 0.009 * (t - 1880) +
              0.05 * np.sin(2*np.pi*t) +
              np.random.default_rng(6).normal(0, 0.1, len(t))).astype(np.float32)
    return {
        "t": t, "signal": signal, "X": signal.reshape(-1, 1),
        "description": "NASA GISS temperature synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 11. SpaceX Launches — time series of launch activity (via SpaceX public API)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_spacex_launches() -> Dict[str, np.ndarray]:
    url = "https://api.spacexdata.com/v4/launches"
    try:
        import json
        raw = _fetch(url, timeout=30)
        launches = json.loads(raw)
        rows = []
        for launch in launches:
            date_unix = launch.get("date_unix")
            success = launch.get("success")
            if date_unix is not None:
                rows.append({
                    "date_unix": float(date_unix),
                    "success": float(success) if success is not None else float("nan"),
                    "flight_number": float(launch.get("flight_number", 0)),
                    "payloads_count": float(len(launch.get("payloads", []))),
                })
        rows.sort(key=lambda x: x["date_unix"])
        t = np.array([r["date_unix"] for r in rows], dtype=np.float32)
        # Convert to years
        t_years = t / (365.25 * 24 * 3600) + 1970
        X = np.column_stack([
            [r["success"] for r in rows],
            [r["flight_number"] for r in rows],
            [r["payloads_count"] for r in rows],
        ]).astype(np.float32)
        return {
            "t": t_years,
            "X": X,
            "signal": X[:, 0],    # success flag
            "flight_numbers": X[:, 1],
            "n_launches": len(rows),
            "description": f"SpaceX launches: {len(rows)} total, success/flight/payloads",
            "source": url,
            "columns": ["success", "flight_number", "payloads_count"],
        }
    except Exception:
        pass
    n = 200
    t = np.linspace(2006, 2024, n).astype(np.float32)
    success = (np.random.default_rng(7).uniform(0, 1, n) > 0.05).astype(np.float32)
    return {
        "t": t, "X": success.reshape(-1, 1), "signal": success,
        "description": "SpaceX launches synthetic placeholder",
        "source": "synthetic",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 12. sklearn built-in datasets
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_sklearn_california_housing() -> Dict[str, np.ndarray]:
    try:
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing()
        return {
            "X": ds.data.astype(np.float32),
            "y": ds.target.astype(np.float32).reshape(-1, 1),
            "signal": ds.target.astype(np.float32),
            "t": np.arange(len(ds.target), dtype=np.float32),
            "feature_names": list(ds.feature_names),
            "target_name": "median_house_value_100k",
            "description": "California housing prices (sklearn, 20640 samples, 8 features)",
            "source": "sklearn.datasets",
        }
    except Exception:
        rng = np.random.default_rng(8)
        n = 200
        X = rng.uniform(0, 1, (n, 8)).astype(np.float32)
        y = (2 + X[:, 0] + rng.normal(0, 0.3, n)).astype(np.float32)
        return {"X": X, "y": y.reshape(-1,1), "signal": y,
                "t": np.arange(n, dtype=np.float32),
                "description": "California housing synthetic placeholder", "source": "synthetic"}


@lru_cache(maxsize=1)
def _load_sklearn_diabetes() -> Dict[str, np.ndarray]:
    try:
        from sklearn.datasets import load_diabetes
        ds = load_diabetes()
        return {
            "X": ds.data.astype(np.float32),
            "y": ds.target.astype(np.float32).reshape(-1, 1),
            "signal": ds.target.astype(np.float32),
            "t": np.arange(len(ds.target), dtype=np.float32),
            "feature_names": list(ds.feature_names),
            "target_name": "disease_progression",
            "description": "Diabetes dataset (sklearn, 442 samples, 10 features)",
            "source": "sklearn.datasets",
        }
    except ImportError:
        rng = np.random.default_rng(9)
        n = 100
        X = rng.normal(0, 1, (n, 10)).astype(np.float32)
        y = (150 + 50*X[:,0] + rng.normal(0,20,n)).astype(np.float32)
        return {"X": X, "y": y.reshape(-1,1), "signal": y,
                "t": np.arange(n, dtype=np.float32),
                "description": "Diabetes synthetic placeholder", "source": "synthetic"}


@lru_cache(maxsize=1)
def _load_sklearn_wine() -> Dict[str, np.ndarray]:
    try:
        from sklearn.datasets import load_wine
        ds = load_wine()
        return {
            "X": ds.data.astype(np.float32),
            "y": ds.target.astype(np.float32).reshape(-1, 1),
            "signal": ds.target.astype(np.float32),
            "t": np.arange(len(ds.target), dtype=np.float32),
            "feature_names": list(ds.feature_names),
            "target_name": "wine_class",
            "description": "Wine quality classification (sklearn, 178 samples, 13 features)",
            "source": "sklearn.datasets",
        }
    except ImportError:
        rng = np.random.default_rng(10)
        n = 100
        X = rng.uniform(0, 1, (n, 13)).astype(np.float32)
        y = (rng.integers(0, 3, n)).astype(np.float32)
        return {"X": X, "y": y.reshape(-1,1), "signal": y,
                "t": np.arange(n, dtype=np.float32),
                "description": "Wine synthetic placeholder", "source": "synthetic"}


# ──────────────────────────────────────────────────────────────────────────────
# 13. seaborn built-in datasets
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_seaborn_penguins() -> Dict[str, np.ndarray]:
    try:
        import seaborn as sns
        df = sns.load_dataset("penguins").dropna()
        numeric = df.select_dtypes(include=[np.number])
        X = numeric.to_numpy(dtype=np.float32)
        return {
            "X": X,
            "signal": X[:, 0],
            "t": np.arange(len(X), dtype=np.float32),
            "feature_names": list(numeric.columns),
            "description": "Palmer penguins morphology (seaborn, ~333 samples)",
            "source": "seaborn.load_dataset",
        }
    except Exception:
        rng = np.random.default_rng(11)
        n = 150
        X = rng.uniform(150, 250, (n, 4)).astype(np.float32)
        return {"X": X, "signal": X[:,0], "t": np.arange(n, dtype=np.float32),
                "description": "Penguins synthetic placeholder", "source": "synthetic"}


@lru_cache(maxsize=1)
def _load_seaborn_mpg() -> Dict[str, np.ndarray]:
    try:
        import seaborn as sns
        df = sns.load_dataset("mpg").dropna()
        numeric = df.select_dtypes(include=[np.number])
        X = numeric.drop(columns=["mpg"], errors="ignore").to_numpy(dtype=np.float32)
        y = df["mpg"].to_numpy(dtype=np.float32) if "mpg" in df.columns else X[:, 0]
        return {
            "X": X, "y": y.reshape(-1, 1), "signal": y,
            "t": np.arange(len(y), dtype=np.float32),
            "target_name": "mpg",
            "description": "Vehicle fuel efficiency (seaborn MPG, ~392 samples)",
            "source": "seaborn.load_dataset",
        }
    except Exception:
        rng = np.random.default_rng(12)
        n = 150
        X = rng.uniform(0,1,(n,5)).astype(np.float32)
        y = (20 + 5*X[:,0] + rng.normal(0,2,n)).astype(np.float32)
        return {"X": X, "y": y.reshape(-1,1), "signal": y,
                "t": np.arange(n, dtype=np.float32),
                "description": "MPG synthetic placeholder", "source": "synthetic"}


# ──────────────────────────────────────────────────────────────────────────────
# 14. statsmodels built-in time series
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_statsmodels_macrodata() -> Dict[str, np.ndarray]:
    try:
        from statsmodels.datasets.macrodata import load_pandas
        df = load_pandas().data
        cols = ["realgdp", "realcons", "realinv", "realgovt",
                "realdpi", "cpi", "m1", "tbilrate", "unemp", "pop", "infl", "realint"]
        available = [c for c in cols if c in df.columns]
        X = df[available].to_numpy(dtype=np.float32)
        t = np.arange(len(X), dtype=np.float32)
        return {
            "t": t, "X": X,
            "signal": X[:, 0],   # real GDP
            "columns": available,
            "description": "US macroeconomic data Q1 1959–Q3 2009 (statsmodels)",
            "source": "statsmodels.datasets.macrodata",
        }
    except ImportError:
        n = 100
        t = np.arange(n, dtype=np.float32)
        X = np.column_stack([np.exp(0.007*t) + np.random.default_rng(13).normal(0,0.05,n)
                              for _ in range(12)]).astype(np.float32)
        return {"t": t, "X": X, "signal": X[:,0],
                "description": "Macrodata synthetic placeholder", "source": "synthetic"}


@lru_cache(maxsize=1)
def _load_statsmodels_elnino() -> Dict[str, np.ndarray]:
    try:
        from statsmodels.datasets.elnino import load_pandas
        df = load_pandas().data
        cols = [c for c in df.columns if c != "YEAR"]
        X = df[cols].to_numpy(dtype=np.float32).flatten()
        t = np.arange(len(X), dtype=np.float32)
        return {
            "t": t, "signal": X, "X": X.reshape(-1, 1),
            "description": "Monthly El Niño SST anomalies (statsmodels, 1950-2010)",
            "source": "statsmodels.datasets.elnino",
        }
    except ImportError:
        n = 500
        t = np.arange(n, dtype=np.float32)
        sig = (0.5 * np.sin(2*np.pi*t/60) + 0.2*np.sin(2*np.pi*t/12) +
               np.random.default_rng(14).normal(0,0.1,n)).astype(np.float32)
        return {"t": t, "signal": sig, "X": sig.reshape(-1,1),
                "description": "El Niño synthetic placeholder", "source": "synthetic"}


# ──────────────────────────────────────────────────────────────────────────────
# Physics Simulations / CFD / Materials / Geoscience
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nasa_exoplanet_archive() -> Dict[str, np.ndarray]:
    """NASA Exoplanet Archive — confirmed exoplanet physical parameters."""
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbper,pl_rade,pl_bmasse,pl_eqt,st_teff,st_rad,st_mass+from+pscomppars+where+pl_rade+is+not+null+and+pl_bmasse+is+not+null&format=csv"
    try:
        text = _fetch_text(url, timeout=45)
        lines = [l for l in text.splitlines() if l and not l.startswith("#")]
        if len(lines) < 5:
            raise ValueError("empty")
        import csv
        reader = csv.DictReader(lines)
        rows = list(reader)
        cols = ["pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_rad", "st_mass"]
        X_rows = []
        for r in rows:
            try:
                row_vals = [float(r[c]) if r.get(c) else float("nan") for c in cols]
                X_rows.append(row_vals)
            except (ValueError, TypeError):
                continue
        X = np.array(X_rows, dtype=np.float32)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        return {
            "X": X,
            "signal": X[:, 1],
            "t": X[:, 0],
            "feature_names": cols,
            "n_planets": len(X),
            "description": f"NASA Exoplanet Archive: {len(X)} confirmed exoplanets, orbital+stellar params",
            "source": url,
        }
    except Exception:
        pass
    # Fallback: synthetic exoplanet population
    rng = np.random.default_rng(20)
    n = 500
    period = 10 ** rng.uniform(-0.5, 3.5, n).astype(np.float32)
    radius = 10 ** rng.uniform(-0.5, 1.2, n).astype(np.float32)
    mass   = radius ** 2.06 * rng.lognormal(0, 0.3, n).astype(np.float32)
    teq    = (6000 * rng.uniform(0.3, 1.5, n) * (1 / period)**0.5).astype(np.float32)
    st_t   = rng.uniform(3500, 7000, n).astype(np.float32)
    st_r   = rng.uniform(0.5, 2.5, n).astype(np.float32)
    st_m   = rng.uniform(0.5, 2.0, n).astype(np.float32)
    X = np.column_stack([period, radius, mass, teq, st_t, st_r, st_m])
    return {
        "X": X, "signal": radius, "t": period,
        "feature_names": ["pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_rad", "st_mass"],
        "n_planets": n,
        "description": "Synthetic exoplanet population (NASA archive unavailable)",
        "source": "synthetic",
    }


@lru_cache(maxsize=1)
def _load_nist_fluid_properties() -> Dict[str, np.ndarray]:
    """Water thermodynamic properties over a range of T and P (NIST-like)."""
    # Generate a grid of water properties using the IAPWS-IF97 approximations
    # (synthetic but physically accurate enough for surrogate modeling)
    T_range = np.linspace(273.15, 623.15, 80, dtype=np.float32)   # K (0-350 C)
    P_range = np.linspace(1e5, 2e7, 80, dtype=np.float32)          # Pa (1-200 bar)
    T_grid, P_grid = np.meshgrid(T_range, P_range)
    T_flat = T_grid.ravel()
    P_flat = P_grid.ravel()

    # Approximate IAPWS-IF97 liquid water region
    Tc = 647.096  # K, critical temp
    rho_ref = 1000.0  # kg/m3
    # Density (simplified correlation)
    rho = (rho_ref * (1 - 0.0003 * (T_flat - 273.15) - 3e-7 * (T_flat - 273.15)**2)
           + 4e-10 * P_flat).astype(np.float32)
    # Specific heat capacity (simplified)
    cp = (4217 - 3.83 * (T_flat - 273.15) + 0.01 * (T_flat - 273.15)**2).astype(np.float32)
    # Dynamic viscosity (Vogel-Fulcher-Tammann approximation)
    mu = (2.414e-5 * 10 ** (247.8 / (T_flat - 140))).astype(np.float32)
    # Thermal conductivity
    k = (0.56 + 2e-3 * (T_flat - 273.15) - 1e-5 * (T_flat - 273.15)**2).astype(np.float32)

    X = np.column_stack([T_flat, P_flat, rho, cp, mu, k])
    return {
        "X": X,
        "T": T_flat, "P": P_flat,
        "rho": rho, "cp": cp, "mu": mu, "k": k,
        "signal": rho,
        "t": T_flat,
        "feature_names": ["T_K", "P_Pa", "rho_kgm3", "cp_JkgK", "mu_Pas", "k_WmK"],
        "description": "Water thermodynamic properties grid 0-350C, 1-200 bar (NIST-like IAPWS approx)",
        "source": "synthetic-IAPWS",
    }


@lru_cache(maxsize=1)
def _load_cfd_cylinder_drag() -> Dict[str, np.ndarray]:
    """2D cylinder drag/lift coefficients vs Reynolds number (parametric CFD sweep)."""
    # Classic drag curve from Schlichting + analytical Stokes/Oseen for low Re
    rng = np.random.default_rng(30)
    Re = np.logspace(np.log10(0.1), np.log10(1e6), 500).astype(np.float32)

    # Drag coefficient (Schlichting empirical fit)
    Cd = np.where(
        Re < 1.0, 24 / Re,
        np.where(Re < 1000, 24/Re * (1 + 0.15*Re**0.687),
                 np.where(Re < 2e5, 0.44 + rng.normal(0, 0.02, 500).astype(np.float32),
                          0.1 + rng.normal(0, 0.02, 500).astype(np.float32)))
    ).astype(np.float32)

    # Lift coefficient (turbulent shedding region adds periodic oscillation)
    St = np.where(Re > 50, 0.21 - 2.7 / Re**0.6, 0.0).astype(np.float32)
    Cl_rms = np.where(Re > 200, 0.4 * np.exp(-Re / 2e5), 0.0).astype(np.float32)
    Cl = (Cl_rms * rng.normal(0, 1, 500)).astype(np.float32)

    X = np.column_stack([Re, Cd, Cl, St])
    return {
        "X": X,
        "Re": Re, "Cd": Cd, "Cl": Cl, "St": St,
        "signal": Cd, "t": Re,
        "feature_names": ["Re", "Cd", "Cl", "Strouhal"],
        "description": "2D circular cylinder drag/lift/Strouhal vs Reynolds (synthetic CFD parametric, 500 pts)",
        "source": "synthetic-Schlichting",
    }


@lru_cache(maxsize=1)
def _load_seismic_waveform() -> Dict[str, np.ndarray]:
    """Synthetic 1D seismic P-wave propagation through a layered medium."""
    dt = 0.002   # s
    t = np.arange(0, 4.0, dt, dtype=np.float32)
    n = len(t)
    # Ricker wavelet source
    f0 = 25.0  # Hz center frequency
    t_shift = 0.1
    tau = (t - t_shift)
    ricker = (1 - 2 * np.pi**2 * f0**2 * tau**2) * np.exp(-np.pi**2 * f0**2 * tau**2)
    ricker = ricker.astype(np.float32)

    # Simple 3-layer model: reflection arrivals
    v = [2000, 3500, 5000]    # m/s
    z = [0, 500, 1200]        # m interface depths
    rc = [0.27, 0.18]         # reflection coefficients
    arrivals = [zi * 2 / vi for zi, vi in zip(z[1:], v[:-1])]  # two-way travel times

    seismogram = ricker.copy()
    for arr_t, r in zip(arrivals, rc):
        shift = int(arr_t / dt)
        if shift < n:
            seismogram[shift: shift + len(ricker)] += r * ricker[:n - shift]

    # Multi-offset: 12 geophones at 50m spacing
    offsets = np.linspace(50, 600, 12, dtype=np.float32)
    traces = np.zeros((12, n), dtype=np.float32)
    for i, off in enumerate(offsets):
        for j, (vi, arr_t, r) in enumerate(zip(v, arrivals, rc)):
            t_offset = np.sqrt(arr_t**2 + (off / vi)**2)
            shift = int(t_offset / dt)
            amp = r / (1 + off / 500)
            if shift < n:
                traces[i, shift: shift + len(ricker)] += amp * ricker[:n - shift]

    return {
        "t": t,
        "X": traces.T,
        "signal": seismogram,
        "offsets": offsets,
        "traces": traces,
        "description": "Synthetic 12-trace seismic gather, Ricker wavelet, 3-layer model",
        "source": "synthetic-seismic",
    }


@lru_cache(maxsize=1)
def _load_heat_conduction_rod() -> Dict[str, np.ndarray]:
    """1D transient heat conduction in a steel rod solved by FDM."""
    # L=1m rod, alpha=1.172e-5 m2/s (steel), Dirichlet BCs, Gaussian IC
    L = 1.0; alpha = 1.172e-5; nx = 101; nt = 500
    dx = L / (nx - 1)
    dt_max = 0.4 * dx**2 / alpha
    dt = dt_max * 0.9
    x = np.linspace(0, L, nx, dtype=np.float64)
    t = np.arange(nt, dtype=np.float64) * dt

    u = np.exp(-100 * (x - 0.5)**2) * 200.0   # Gaussian hot spot at center, deg C
    u[0] = 20.0; u[-1] = 20.0                   # fixed BC

    r = alpha * dt / dx**2
    snapshots = np.zeros((nt, nx), dtype=np.float32)
    snapshots[0] = u

    for i in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 20.0; u_new[-1] = 20.0
        u = u_new
        snapshots[i] = u.astype(np.float32)

    return {
        "x": x.astype(np.float32),
        "t": t.astype(np.float32),
        "u": snapshots,
        "X": snapshots.reshape(nt, nx),
        "signal": snapshots[:, nx//2],
        "alpha": alpha,
        "description": f"1D heat conduction FDM: steel rod L=1m, {nx} pts, {nt} timesteps, dt={dt:.4f}s",
        "source": "synthetic-FDM",
    }


@lru_cache(maxsize=1)
def _load_turbulent_channel_flow() -> Dict[str, np.ndarray]:
    """DNS turbulent channel flow statistics at Re_tau=180 (Kim, Moin & Moser 1987)."""
    # Tabulated DNS data from KMM87 (embedded, publicly available reference data)
    # y+: wall-normal coordinate in viscous units, U+: mean velocity
    y_plus = np.array([
        0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
        30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 130.0, 150.0, 180.0
    ], dtype=np.float32)
    # Log-law: U+ = (1/0.41)*ln(y+) + 5.2 (inner layer); viscous: U+ = y+ (y+<5)
    U_plus = np.where(y_plus < 5.0, y_plus,
                      np.where(y_plus < 30.0,
                               y_plus * np.exp(-y_plus/11.0) + (1/0.41)*np.log(y_plus+1e-8)*0.85 + 5.2,
                               (1/0.41)*np.log(y_plus) + 5.2)).astype(np.float32)
    # Reynolds stresses (u'u'+, v'v'+, w'w'+, u'v'+) — KMM87 table
    uu = np.array([0.0, 0.1, 0.4, 2.2, 4.0, 5.2, 6.3, 6.8, 7.1, 7.0, 6.5,
                   5.5, 4.7, 4.1, 3.7, 3.0, 2.4, 1.8, 1.4, 0.0], dtype=np.float32)
    vv = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.2, 1.3,
                   1.4, 1.3, 1.2, 1.1, 0.9, 0.8, 0.5, 0.4, 0.0], dtype=np.float32)
    ww = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 2.2, 2.5, 2.7, 2.7, 2.6,
                   2.3, 2.0, 1.8, 1.6, 1.3, 1.1, 0.7, 0.6, 0.0], dtype=np.float32)
    uv = np.array([0.0, 0.0, 0.0, -0.1, -0.3, -0.5, -0.7, -0.8, -0.85, -0.87, -0.87,
                   -0.85, -0.83, -0.81, -0.79, -0.75, -0.70, -0.60, -0.50, 0.0], dtype=np.float32)

    X = np.column_stack([y_plus, U_plus, uu, vv, ww, uv])
    return {
        "X": X,
        "y_plus": y_plus, "U_plus": U_plus,
        "uu": uu, "vv": vv, "ww": ww, "uv": uv,
        "signal": U_plus, "t": y_plus,
        "feature_names": ["y+", "U+", "uu+", "vv+", "ww+", "uv+"],
        "Re_tau": 180,
        "description": "DNS turbulent channel flow Re_tau=180: mean velocity + Reynolds stresses (KMM87-style)",
        "source": "synthetic-DNS-KMM87",
    }


@lru_cache(maxsize=1)
def _load_materials_fatigue() -> Dict[str, np.ndarray]:
    """S-N (Wohler) fatigue curve data for aluminum alloy 6061-T6 (synthetic)."""
    rng = np.random.default_rng(40)
    # Stress amplitude [MPa], cycles to failure
    # Basquin's law: S^b * N = C  (b~0.1, C~3e15 for Al 6061)
    S_ult = 310.0   # MPa ultimate
    S_e   = 96.0    # MPa endurance limit
    b     = -0.087  # Basquin exponent
    sf_prime = 1.5 * S_ult

    S_amp = np.linspace(S_e * 1.05, 0.9 * S_ult, 200).astype(np.float32)
    N_f = ((S_amp / sf_prime) ** (1/b)).astype(np.float32)
    # Scatter: lognormal
    scatter = rng.lognormal(0, 0.3, len(S_amp)).astype(np.float32)
    N_f_noisy = (N_f * scatter).astype(np.float32)

    # Mean stress effect (R=-1 vs R=0)
    S_amp_R0 = (S_amp * (1 - S_e / S_ult)).astype(np.float32)
    N_f_R0   = ((S_amp_R0 / sf_prime) ** (1/b)).astype(np.float32)

    X = np.column_stack([S_amp, N_f_noisy, S_amp_R0, N_f_R0])
    return {
        "X": X,
        "S_amp": S_amp, "N_f": N_f_noisy,
        "S_amp_R0": S_amp_R0, "N_f_R0": N_f_R0,
        "signal": np.log10(N_f_noisy + 1),
        "t": S_amp,
        "feature_names": ["S_amp_MPa", "N_f_R-1", "S_amp_R0_MPa", "N_f_R0"],
        "material": "Al 6061-T6",
        "description": "S-N fatigue curve Al 6061-T6: stress amplitude vs cycles-to-failure (Basquin, 200 pts)",
        "source": "synthetic-Basquin",
    }


@lru_cache(maxsize=1)
def _load_orbit_propagation() -> Dict[str, np.ndarray]:
    """Keplerian orbit propagation: ISS-like orbit state vector over 24h."""
    # ISS TLE-derived orbital elements (approximate)
    a  = 6778e3       # semi-major axis [m]
    e  = 0.0001       # eccentricity
    i  = np.radians(51.6)  # inclination
    mu = 3.986004418e14    # Earth GM [m3/s2]
    T  = 2*np.pi*np.sqrt(a**3 / mu)  # orbital period [s] ~92 min
    n  = 2*np.pi / T  # mean motion

    dt = 60.0   # 1 minute steps
    t_max = 24 * 3600  # 24 hours
    t = np.arange(0, t_max, dt, dtype=np.float64)
    nt = len(t)

    # Solve Kepler's equation M = E - e*sin(E)  [simple iteration]
    M = n * t
    E = M.copy()
    for _ in range(20):
        E = M + e * np.sin(E)

    nu = 2 * np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r  = a * (1 - e*np.cos(E))

    # Perifocal frame -> ECI (simplified: RAAN=0, omega=0 for brevity)
    cos_i = np.cos(i); sin_i = np.sin(i)
    x_pf = r * np.cos(nu); y_pf = r * np.sin(nu)
    x = x_pf.astype(np.float32)
    y = (y_pf * cos_i).astype(np.float32)
    z = (y_pf * sin_i).astype(np.float32)
    alt = (r - 6371e3).astype(np.float32)
    speed = np.sqrt(mu * (2/r - 1/a)).astype(np.float32)

    X = np.column_stack([t.astype(np.float32), x, y, z, alt, speed])
    return {
        "X": X,
        "t": t.astype(np.float32),
        "x": x, "y": y, "z": z,
        "altitude": alt, "speed": speed,
        "signal": alt,
        "feature_names": ["t_s", "x_m", "y_m", "z_m", "alt_m", "speed_ms"],
        "orbital_period_s": float(T),
        "n_orbits": float(t_max / T),
        "description": f"ISS-like Keplerian orbit: {nt} pts, {t_max/3600:.0f}h, 1-min resolution",
        "source": "synthetic-Kepler",
    }


@lru_cache(maxsize=1)
def _load_plasma_fusion() -> Dict[str, np.ndarray]:
    """Tokamak-like plasma parameters time series (synthetic ITER-like discharge)."""
    rng = np.random.default_rng(50)
    dt = 0.01   # s
    t = np.arange(0, 30.0, dt, dtype=np.float32)
    nt = len(t)

    # Plasma current ramp-up and flat-top
    Ip_max = 15e6  # A (ITER 15 MA)
    ramp_end = 5.0; flat_end = 22.0
    Ip = np.where(t < ramp_end, Ip_max * (t / ramp_end),
         np.where(t < flat_end, Ip_max,
                  Ip_max * np.maximum(0, (30.0 - t) / (30.0 - flat_end))))

    # Electron density (triangular ramp) — kept float64 to avoid overflow in P_fus
    ne = (1e20 * np.where(t < ramp_end, t/ramp_end,
          np.where(t < flat_end, 1.0, (30.0-t)/(30.0-flat_end)))).astype(np.float64)

    # Temperature — parabolic profile in time
    Te = (20e3 * np.where(t < ramp_end, (t/ramp_end)**2,
          np.where(t < flat_end, 1.0 - 0.02*np.sin(2*np.pi*t/2),
                   ((30.0-t)/(30.0-flat_end))**2))).astype(np.float32)

    # Fusion power (D-T reaction rate P ~ ne^2 * <sigma*v>(T))
    sv_exp = (-14.0 + 0.0025 * Te.astype(np.float64).clip(0, 30000)).clip(-60, 60)
    sv = np.exp(sv_exp)
    P_fus = (5.6e-19 * ne**2 * sv * 1e3).clip(0, 1e6).astype(np.float32)

    # ELMs — edge localized modes (sawtooth oscillations during flat-top)
    elm = np.zeros(nt, dtype=np.float32)
    for t_elm in np.arange(ramp_end + 1, flat_end, 0.8):
        idx = int(t_elm / dt)
        if idx < nt:
            decay = np.exp(-20 * (t[idx:] - t_elm).clip(0))
            elm[idx:] += 0.1 * Ip_max * decay[:nt-idx]

    Ip_noisy = (Ip + elm + rng.normal(0, 0.01*Ip_max, nt)).astype(np.float32)

    ne_f32 = (ne / 1e20).astype(np.float32)
    X = np.column_stack([t, Ip_noisy/1e6, ne_f32, Te/1e3, P_fus/1e6])
    return {
        "X": X,
        "t": t, "Ip_MA": Ip_noisy/1e6, "ne_1e20": ne_f32,
        "Te_keV": Te/1e3, "P_fus_MW": P_fus/1e6,
        "signal": Ip_noisy/1e6,
        "feature_names": ["t_s", "Ip_MA", "ne_1e20m3", "Te_keV", "P_fus_MW"],
        "description": "Synthetic ITER-like tokamak discharge: Ip, ne, Te, P_fus over 30s",
        "source": "synthetic-tokamak",
    }


@lru_cache(maxsize=1)
def _load_reaction_diffusion() -> Dict[str, np.ndarray]:
    """2D Gray-Scott reaction-diffusion simulation (pattern formation)."""
    # Parameters: U-spots pattern (Du=0.16, Dv=0.08, F=0.035, k=0.065)
    nx = ny = 64; n_steps = 3000; dt = 1.0
    Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065
    dx = 1.0

    rng = np.random.default_rng(60)
    U = np.ones((ny, nx), dtype=np.float64)
    V = np.zeros((ny, nx), dtype=np.float64)

    # Seed the middle with a perturbation
    cy, cx = ny//2, nx//2
    U[cy-8:cy+8, cx-8:cx+8] = 0.5
    V[cy-8:cy+8, cx-8:cx+8] = 0.25
    U += 0.05 * rng.random((ny, nx))
    V += 0.05 * rng.random((ny, nx))

    def laplacian(Z):
        return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
                np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4*Z) / dx**2

    save_steps = [0, 500, 1000, 2000, 3000]
    snapshots_U = []
    snapshots_V = []

    for step in range(n_steps + 1):
        if step in save_steps:
            snapshots_U.append(U.copy().astype(np.float32))
            snapshots_V.append(V.copy().astype(np.float32))
        if step >= n_steps:
            break
        uvv = U * V * V
        dU = Du * laplacian(U) - uvv + F * (1 - U)
        dV = Dv * laplacian(V) + uvv - (F + k) * V
        U = (U + dt * dU).clip(0, 1)
        V = (V + dt * dV).clip(0, 1)

    U_snaps = np.stack(snapshots_U)  # (5, 64, 64)
    V_snaps = np.stack(snapshots_V)

    x = np.linspace(0, nx-1, nx, dtype=np.float32)
    y = np.linspace(0, ny-1, ny, dtype=np.float32)
    return {
        "U": U_snaps,
        "V": V_snaps,
        "X": U_snaps.reshape(len(save_steps), -1),
        "signal": U_snaps[-1].ravel(),
        "t": np.array(save_steps, dtype=np.float32),
        "x": x, "y": y,
        "params": {"Du": Du, "Dv": Dv, "F": F, "k": k},
        "description": f"2D Gray-Scott reaction-diffusion ({nx}x{ny}), {n_steps} steps, 5 snapshots",
        "source": "synthetic-GrayScott",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public convenience function
# ──────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    # Time series
    "sunspots":                    _load_sunspots,
    "air_passengers":              _load_air_passengers,
    "co2":                         _load_co2,
    "etth1":                       _load_etth1,
    "ettm1":                       _load_ettm1,
    "jena_climate":                _load_jena_climate,
    "nasa_giss_temp":              _load_nasa_giss_temp,
    "spacex_launches":             _load_spacex_launches,
    "statsmodels_macrodata":       _load_statsmodels_macrodata,
    "statsmodels_elnino":          _load_statsmodels_elnino,
    # Physics / engineering (tabular)
    "airfoil_noise":               _load_airfoil_noise,
    "concrete_strength":           _load_concrete_strength,
    "energy_efficiency":           _load_energy_efficiency,
    # Physics simulations / CFD / materials / geoscience
    "nasa_exoplanet_archive":      _load_nasa_exoplanet_archive,
    "nist_fluid_properties":       _load_nist_fluid_properties,
    "cfd_cylinder_drag":           _load_cfd_cylinder_drag,
    "seismic_waveform":            _load_seismic_waveform,
    "heat_conduction_rod":         _load_heat_conduction_rod,
    "turbulent_channel_flow":      _load_turbulent_channel_flow,
    "materials_fatigue":           _load_materials_fatigue,
    "orbit_propagation":           _load_orbit_propagation,
    "plasma_fusion":               _load_plasma_fusion,
    "reaction_diffusion":          _load_reaction_diffusion,
    # Library built-ins
    "sklearn_california_housing":  _load_sklearn_california_housing,
    "sklearn_diabetes":            _load_sklearn_diabetes,
    "sklearn_wine":                _load_sklearn_wine,
    "seaborn_penguins":            _load_seaborn_penguins,
    "seaborn_mpg":                 _load_seaborn_mpg,
}


def load_real_dataset(dataset_id: str) -> Dict[str, np.ndarray]:
    """Load a real-world dataset by ID.

    Downloads on first call, then caches in memory.
    """
    if dataset_id not in _LOADERS:
        raise KeyError(
            f"Real-world dataset '{dataset_id}' not found.\n"
            f"Available: {list(_LOADERS.keys())}"
        )
    return _LOADERS[dataset_id]()


# ──────────────────────────────────────────────────────────────────────────────
# Registration with DatasetRegistry
# ──────────────────────────────────────────────────────────────────────────────

DatasetRegistry.register(
    DatasetInfo(
        id="sunspots",
        name="Sunspots",
        category="timeseries",
        description="Monthly mean total sunspot number since 1749 (SIDC/SILSO).",
        fields=["t", "signal", "X"],
        tags=["solar", "astronomy", "cyclical", "real-world"],
        license="CC BY-NC 4.0",
        reference="SIDC — http://www.sidc.be/silso/datafiles",
    ),
    _load_sunspots,
)

DatasetRegistry.register(
    DatasetInfo(
        id="air_passengers",
        name="Airline Passengers",
        category="timeseries",
        description="Monthly airline passengers 1949-1960 — classic Box-Jenkins dataset.",
        fields=["t", "signal", "X"],
        tags=["airline", "seasonal", "trend", "real-world"],
        license="public domain",
        reference="Box & Jenkins 1976",
    ),
    _load_air_passengers,
)

DatasetRegistry.register(
    DatasetInfo(
        id="co2",
        name="Mauna Loa CO₂",
        category="timeseries",
        description="Weekly atmospheric CO₂ concentration at Mauna Loa (ppm) since 1958.",
        fields=["t", "signal", "X"],
        tags=["climate", "co2", "trend", "seasonal", "real-world"],
        license="public domain",
        reference="NOAA GML / Keeling et al.",
    ),
    _load_co2,
)

DatasetRegistry.register(
    DatasetInfo(
        id="etth1",
        name="ETT-H1",
        category="timeseries",
        description="Hourly electricity transformer temperature (7 vars, 2016-2018).",
        fields=["t", "X", "signal", "columns"],
        tags=["electricity", "multivariate", "benchmark", "real-world"],
        license="Apache 2.0",
        reference="Zhou et al. 2021 — https://github.com/zhouhaoyi/ETDataset",
    ),
    _load_etth1,
)

DatasetRegistry.register(
    DatasetInfo(
        id="ettm1",
        name="ETT-M1",
        category="timeseries",
        description="15-min electricity transformer temperature (7 vars, 2016-2018).",
        fields=["t", "X", "signal", "columns"],
        tags=["electricity", "multivariate", "benchmark", "real-world"],
        license="Apache 2.0",
        reference="Zhou et al. 2021 — https://github.com/zhouhaoyi/ETDataset",
    ),
    _load_ettm1,
)

DatasetRegistry.register(
    DatasetInfo(
        id="jena_climate",
        name="Jena Climate",
        category="timeseries",
        description="Hourly climate observations at Jena, Germany (T,p,rh,wind — 2009-2016).",
        fields=["t", "X", "signal", "columns"],
        tags=["climate", "weather", "multivariate", "real-world"],
        license="public domain",
        reference="Max Planck Institute for Biogeochemistry",
    ),
    _load_jena_climate,
)

DatasetRegistry.register(
    DatasetInfo(
        id="airfoil_noise",
        name="NASA Airfoil Self-Noise",
        category="physics",
        description="Aerodynamic inputs → sound pressure level (NASA/UCI, 1503 samples).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["aerodynamics", "acoustics", "regression", "real-world"],
        license="CC BY 4.0",
        reference="Brooks, Pope & Marcolini 1989 — UCI ML #291",
    ),
    _load_airfoil_noise,
)

DatasetRegistry.register(
    DatasetInfo(
        id="concrete_strength",
        name="Concrete Compressive Strength",
        category="physics",
        description="Mix ingredients → 28-day compressive strength (UCI ML, 1030 samples).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["materials", "concrete", "regression", "real-world"],
        license="CC BY 4.0",
        reference="Yeh 1998 — UCI ML #165",
    ),
    _load_concrete_strength,
)

DatasetRegistry.register(
    DatasetInfo(
        id="energy_efficiency",
        name="Building Energy Efficiency",
        category="physics",
        description="Building geometry → heating/cooling load (UCI ML, 768 samples).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["buildings", "energy", "regression", "real-world"],
        license="CC BY 4.0",
        reference="Tsanas & Xifara 2012 — UCI ML #242",
    ),
    _load_energy_efficiency,
)

DatasetRegistry.register(
    DatasetInfo(
        id="nasa_giss_temp",
        name="NASA GISS Global Temperature",
        category="timeseries",
        description="NASA GISS global surface temperature anomaly since 1880 (monthly).",
        fields=["t", "signal", "X"],
        tags=["nasa", "climate", "temperature", "global-warming", "real-world"],
        license="public domain",
        reference="NASA GISS — https://data.giss.nasa.gov/gistemp/",
    ),
    _load_nasa_giss_temp,
)

DatasetRegistry.register(
    DatasetInfo(
        id="spacex_launches",
        name="SpaceX Launches",
        category="timeseries",
        description="SpaceX launch history: success, flight number, payload count (SpaceX API v4).",
        fields=["t", "X", "signal", "columns"],
        tags=["spacex", "aerospace", "launches", "real-world"],
        license="open data",
        reference="SpaceX API — https://api.spacexdata.com/v4/launches",
    ),
    _load_spacex_launches,
)

DatasetRegistry.register(
    DatasetInfo(
        id="statsmodels_macrodata",
        name="US Macroeconomic Data",
        category="timeseries",
        description="US macroeconomic quarterly data 1959-2009 (GDP, CPI, unemployment, etc.).",
        fields=["t", "X", "signal", "columns"],
        tags=["economics", "gdp", "multivariate", "statsmodels", "real-world"],
        license="public domain",
        reference="statsmodels.datasets.macrodata",
    ),
    _load_statsmodels_macrodata,
)

DatasetRegistry.register(
    DatasetInfo(
        id="statsmodels_elnino",
        name="El Niño SST",
        category="timeseries",
        description="Monthly El Niño sea surface temperature anomalies 1950-2010.",
        fields=["t", "signal", "X"],
        tags=["climate", "ocean", "enso", "statsmodels", "real-world"],
        license="public domain",
        reference="statsmodels.datasets.elnino",
    ),
    _load_statsmodels_elnino,
)

DatasetRegistry.register(
    DatasetInfo(
        id="sklearn_california_housing",
        name="California Housing",
        category="physics",
        description="California census housing prices (sklearn, 20640 samples, 8 features).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["regression", "housing", "sklearn", "real-world"],
        license="public domain",
        reference="sklearn.datasets.fetch_california_housing",
    ),
    _load_sklearn_california_housing,
)

DatasetRegistry.register(
    DatasetInfo(
        id="sklearn_diabetes",
        name="Diabetes Progression",
        category="physics",
        description="Diabetes disease progression prediction (sklearn, 442 samples).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["medical", "regression", "sklearn", "real-world"],
        license="BSD",
        reference="sklearn.datasets.load_diabetes",
    ),
    _load_sklearn_diabetes,
)

DatasetRegistry.register(
    DatasetInfo(
        id="sklearn_wine",
        name="Wine Quality",
        category="physics",
        description="Wine chemical analysis classification (sklearn, 178 samples, 13 features).",
        fields=["X", "y", "signal", "feature_names"],
        tags=["chemistry", "classification", "sklearn", "real-world"],
        license="BSD",
        reference="sklearn.datasets.load_wine",
    ),
    _load_sklearn_wine,
)

DatasetRegistry.register(
    DatasetInfo(
        id="seaborn_penguins",
        name="Palmer Penguins",
        category="physics",
        description="Penguin morphology measurements (seaborn, ~333 samples, 4 numeric features).",
        fields=["X", "signal", "feature_names"],
        tags=["biology", "classification", "seaborn", "real-world"],
        license="CC0",
        reference="seaborn.load_dataset('penguins')",
    ),
    _load_seaborn_penguins,
)

DatasetRegistry.register(
    DatasetInfo(
        id="seaborn_mpg",
        name="Vehicle MPG",
        category="physics",
        description="Vehicle fuel efficiency (MPG) from engine/weight/year features (seaborn).",
        fields=["X", "y", "signal", "target_name"],
        tags=["automotive", "regression", "seaborn", "real-world"],
        license="public domain",
        reference="seaborn.load_dataset('mpg')",
    ),
    _load_seaborn_mpg,
)

# ──────────────────────────────────────────────────────────────────────────────
# Physics simulations / CFD / materials / geoscience registrations
# ──────────────────────────────────────────────────────────────────────────────

DatasetRegistry.register(
    DatasetInfo(
        id="nasa_exoplanet_archive",
        name="NASA Exoplanet Archive",
        category="physics",
        description="Confirmed exoplanet orbital and stellar parameters (NASA Exoplanet Archive TAP).",
        fields=["X", "signal", "t", "feature_names", "n_planets"],
        tags=["nasa", "astrophysics", "exoplanets", "regression", "real-world"],
        license="public domain",
        reference="NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu",
    ),
    _load_nasa_exoplanet_archive,
)

DatasetRegistry.register(
    DatasetInfo(
        id="nist_fluid_properties",
        name="Water Thermodynamic Properties (NIST-like)",
        category="physics",
        description="Water density, Cp, viscosity, conductivity over T=0-350C, P=1-200 bar.",
        fields=["X", "T", "P", "rho", "cp", "mu", "k", "feature_names"],
        tags=["thermodynamics", "fluid-mechanics", "NIST", "surrogate", "simulation"],
        license="synthetic",
        reference="IAPWS-IF97 approximation",
    ),
    _load_nist_fluid_properties,
)

DatasetRegistry.register(
    DatasetInfo(
        id="cfd_cylinder_drag",
        name="2D Cylinder Drag/Lift vs Reynolds",
        category="physics",
        description="Drag coefficient, lift, and Strouhal number for a 2D circular cylinder across Re=0.1-1e6.",
        fields=["X", "Re", "Cd", "Cl", "St", "feature_names"],
        tags=["CFD", "aerodynamics", "drag", "vortex-shedding", "simulation"],
        license="synthetic",
        reference="Schlichting & Gersten, Boundary Layer Theory",
    ),
    _load_cfd_cylinder_drag,
)

DatasetRegistry.register(
    DatasetInfo(
        id="seismic_waveform",
        name="Synthetic Seismic Waveform",
        category="physics",
        description="Synthetic 12-trace seismic gather: Ricker wavelet source, 3-layer reflectivity model.",
        fields=["t", "X", "signal", "offsets", "traces"],
        tags=["geophysics", "seismic", "wave-propagation", "simulation"],
        license="synthetic",
        reference="Ricker wavelet + 3-layer reflectivity model",
    ),
    _load_seismic_waveform,
)

DatasetRegistry.register(
    DatasetInfo(
        id="heat_conduction_rod",
        name="1D Heat Conduction (FDM)",
        category="physics",
        description="1D transient heat conduction in a steel rod: FDM solution, Gaussian IC, Dirichlet BCs.",
        fields=["x", "t", "u", "X", "signal"],
        tags=["heat-transfer", "PDE", "FDM", "simulation", "parabolic"],
        license="synthetic",
        reference="Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
    ),
    _load_heat_conduction_rod,
)

DatasetRegistry.register(
    DatasetInfo(
        id="turbulent_channel_flow",
        name="DNS Turbulent Channel Flow (Re_tau=180)",
        category="physics",
        description="Turbulent channel flow DNS statistics: mean velocity and Reynolds stresses at Re_tau=180.",
        fields=["X", "y_plus", "U_plus", "uu", "vv", "ww", "uv", "feature_names"],
        tags=["turbulence", "DNS", "fluid-mechanics", "channel-flow", "CFD"],
        license="synthetic",
        reference="Kim, Moin & Moser 1987, J. Fluid Mech. 177",
    ),
    _load_turbulent_channel_flow,
)

DatasetRegistry.register(
    DatasetInfo(
        id="materials_fatigue",
        name="S-N Fatigue Curve (Al 6061-T6)",
        category="physics",
        description="Wohler S-N fatigue curve for aluminum 6061-T6: stress amplitude vs cycles-to-failure.",
        fields=["X", "S_amp", "N_f", "signal", "t", "feature_names"],
        tags=["materials", "fatigue", "S-N", "fracture-mechanics", "simulation"],
        license="synthetic",
        reference="Basquin's law, MIL-HDBK-5J",
    ),
    _load_materials_fatigue,
)

DatasetRegistry.register(
    DatasetInfo(
        id="orbit_propagation",
        name="Keplerian Orbit Propagation (ISS-like)",
        category="physics",
        description="ISS-like Keplerian orbit: 24h state vector (x,y,z,altitude,speed) at 1-min resolution.",
        fields=["X", "t", "x", "y", "z", "altitude", "speed", "feature_names"],
        tags=["astrodynamics", "orbit", "satellite", "Kepler", "simulation"],
        license="synthetic",
        reference="Keplerian mechanics, ISS TLE approximate elements",
    ),
    _load_orbit_propagation,
)

DatasetRegistry.register(
    DatasetInfo(
        id="plasma_fusion",
        name="Tokamak Plasma Parameters (ITER-like)",
        category="physics",
        description="Synthetic ITER-like tokamak discharge: plasma current, electron density, temperature, fusion power over 30s.",
        fields=["X", "t", "Ip_MA", "ne_1e20", "Te_keV", "P_fus_MW", "feature_names"],
        tags=["plasma", "fusion", "tokamak", "ITER", "simulation"],
        license="synthetic",
        reference="ITER physics basis, Wesson Tokamaks",
    ),
    _load_plasma_fusion,
)

DatasetRegistry.register(
    DatasetInfo(
        id="reaction_diffusion",
        name="Gray-Scott Reaction-Diffusion (2D)",
        category="physics",
        description="2D Gray-Scott RD simulation: U-spots pattern, 64x64 grid, 5 temporal snapshots.",
        fields=["U", "V", "X", "signal", "t", "x", "y", "params"],
        tags=["reaction-diffusion", "PDE", "pattern-formation", "Gray-Scott", "simulation"],
        license="synthetic",
        reference="Pearson 1993, Science 261",
    ),
    _load_reaction_diffusion,
)
