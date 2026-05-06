"""
NASA POWER (Daily) -> UPD (PhysicalSample) -> Zarr
(+ optional pinneaple_pdb ingestion if available)

Ref:
  /api/temporal/daily/point?parameters=...&community=...&longitude=...&latitude=...&start=...&end=...&format=JSON
  (NASA POWER Daily API docs)
"""

from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Dict, Any, List

import torch

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore


def fetch_power_daily(
    *,
    latitude: float,
    longitude: float,
    start: str,   # "YYYYMMDD"
    end: str,     # "YYYYMMDD"
    parameters: List[str],
    community: str = "SB",
    time_standard: str = "UTC",
) -> Dict[str, Any]:
    """
    Fetch NASA POWER daily point data (JSON).
    No API key required for POWER.

    Returns parsed JSON dict.
    """
    import requests

    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": str(float(longitude)),
        "latitude": str(float(latitude)),
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": time_standard,
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def power_json_to_samples(power: Dict[str, Any]) -> List[PhysicalSample]:
    """
    Convert POWER response to a list of PhysicalSample.
    MVP: 1 sample with a time-series tensor.

    Output contract:
      state["x"] : (T, D) float32
      state["t"] : (T, 1) float32  (day index)
      schema: includes parameter names + units if present
    """
    props = power.get("properties", {})
    param = props.get("parameter", {})  # dict: {PARAM: {YYYYMMDD: value}}
    if not param:
        raise ValueError("POWER response missing properties.parameter")

    # Sort dates using the first parameter as reference
    first_k = sorted(param.keys())[0]
    dates = sorted(param[first_k].keys())

    keys = sorted(param.keys())
    T = len(dates)
    D = len(keys)

    x = torch.empty((T, D), dtype=torch.float32)
    for j, k in enumerate(keys):
        series = param[k]
        for i, d in enumerate(dates):
            v = series.get(d, float("nan"))
            x[i, j] = float(v) if v is not None else float("nan")

    # simple time coordinate: 0..T-1
    t = torch.arange(T, dtype=torch.float32).unsqueeze(1)

    # metadata / provenance
    header = power.get("header", {})
    meta = {
        "source": "nasa_power",
        "title": header.get("title"),
        "api": header.get("api"),
        "start": props.get("start"),
        "end": props.get("end"),
        "parameters": keys,
        "dates": dates[:10] + (["..."] if len(dates) > 10 else []),
    }

    # units if present
    # Some POWER responses include parameter metadata in "parameters" elsewhere; keep MVP robust:
    units = {k: "POWER_unit_unknown" for k in keys}

    s = PhysicalSample(
        state={"x": x, "t": t},
        domain={"type": "point", "crs": "WGS84"},
        provenance=meta,
        schema={"units": units, "columns": keys, "time": "daily"},
    )
    return [s]

def main():
    out_dir = "examples/_out/pdb_nasa_power"
    os.makedirs(out_dir, exist_ok=True)

    zarr_path = os.path.join(out_dir, "nasa_power_daily_point.zarr")

    # Example: Recife-ish lat/lon (adjust as you want)
    latitude = -8.05
    longitude = -34.9

    # Choose a small window for demo
    start = "20240101"
    end = "20240131"

    parameters = ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]

    power = fetch_power_daily(
        latitude=latitude,
        longitude=longitude,
        start=start,
        end=end,
        parameters=parameters,
        community="SB",
        time_standard="UTC",
    )

    samples = power_json_to_samples(power)

    # Write UPD -> Zarr
    UPDZarrStore.write(
        zarr_path,
        samples,
        manifest={
            "name": "nasa_power_daily_point",
            "source": "NASA POWER Daily API",
            "latitude": latitude,
            "longitude": longitude,
            "start": start,
            "end": end,
            "parameters": parameters,
        },
    )

    print("wrote:", zarr_path)
    
    # save raw for inspection
    raw_path = os.path.join(out_dir, "raw_power.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(power, f, indent=2)
    print("saved raw:", raw_path)


if __name__ == "__main__":
    main()
