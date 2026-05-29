"""Sharding and subsetting of physical datasets by time and spatial tiles."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

@dataclass
class ShardSpec:
    """Specification for time-window and optional spatial tiling of datasets."""

    time_window: str = "6H"  # pandas offset alias
    tile_deg: Optional[Tuple[float, float]] = None  # (dlat, dlon) e.g. (30, 30)
    add_regime_tags: bool = True     # MVP: only tags in metadata

def iter_time_windows(
    time_values, window: str
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (start, end) time pairs for non-overlapping windows over time_values."""
    t = pd.to_datetime(time_values)
    if len(t) == 0:
        return
    start = t.min()
    end = t.max()
    edges = pd.date_range(start=start.floor(window), end=end.ceil(window), freq=window)
    if len(edges) < 2:
        edges = pd.DatetimeIndex([start, end])
    for a, b in zip(edges[:-1], edges[1:]):
        yield a, b

def subset_time(ds: xr.Dataset, a: pd.Timestamp, b: pd.Timestamp) -> xr.Dataset:
    """Select time slice [a, b) from dataset."""
    # include a, exclude b
    b2 = b - pd.Timedelta(nanoseconds=1)
    return ds.sel(time=slice(a.to_datetime64(), b2.to_datetime64()))

def iter_tiles(lat_vals, lon_vals, dlat: float, dlon: float):
    """Yield (lat0, lat1, lon0, lon1) tile bounds aligned to dlat/dlon grid."""
    lat_vals = np.asarray(lat_vals, dtype=float)
    lon_vals = np.asarray(lon_vals, dtype=float)
    lat_min, lat_max = float(np.min(lat_vals)), float(np.max(lat_vals))
    lon_min, lon_max = float(np.min(lon_vals)), float(np.max(lon_vals))

    lat_edges = np.arange(np.floor(lat_min / dlat) * dlat, lat_max + dlat, dlat)
    lon_edges = np.arange(np.floor(lon_min / dlon) * dlon, lon_max + dlon, dlon)

    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            yield (lat_edges[i], lat_edges[i + 1], lon_edges[j], lon_edges[j + 1])

def subset_tile(
    ds: xr.Dataset, lat0: float, lat1: float, lon0: float, lon1: float
) -> xr.Dataset:
    """Select spatial tile [lat0,lat1] x [lon0,lon1]; assumes coords sorted."""
    # assumes coords sorted
    return ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))

def regime_tags_for(ds: xr.Dataset) -> list[str]:
    """Derive regime tags (e.g. tropics, polar) from dataset lat extent."""
    tags = []
    if "lat" in ds.coords:
        lat = ds["lat"].values
        # tropics if the tile intersects the tropics band
        if float(np.min(lat)) < 23.5 and float(np.max(lat)) > -23.5:
            tags.append("tropics")
        if float(np.min(lat)) > 60 or float(np.max(lat)) < -60:
            tags.append("polar")
    return tags
