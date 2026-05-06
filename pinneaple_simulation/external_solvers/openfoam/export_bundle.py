from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil
import pandas as pd


def export_bundle(
    *,
    project_dir: Path,
    bundle_dir: Path,
) -> None:
    """
    Creates bundle_dir with:
      bundle/geometry.stl, bundle/manifest.json, bundle/conditions.json, bundle/sensors.parquet
      derived/points_*.parquet, derived/line_points.parquet, derived/sensors_points.parquet
    """
    project_dir = project_dir.resolve()
    bundle_dir = bundle_dir.resolve()

    (bundle_dir / "bundle").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "derived").mkdir(parents=True, exist_ok=True)

    # geometry
    shutil.copy(project_dir / "geometry" / "channel_minus_cylinder.stl", bundle_dir / "bundle" / "geometry.stl")

    # specs
    shutil.copy(project_dir / "specs" / "manifest.json", bundle_dir / "bundle" / "manifest.json")
    shutil.copy(project_dir / "specs" / "conditions.json", bundle_dir / "bundle" / "conditions.json")

    # derived
    for f in ["points_collocation.parquet", "points_boundary.parquet", "sensors_points.parquet", "line_points.parquet"]:
        shutil.copy(project_dir / "derived" / f, bundle_dir / "derived" / f)

    # sensors (OpenFOAM sampled)
    shutil.copy(project_dir / "datasets" / "sensors.parquet", bundle_dir / "bundle" / "sensors.parquet")
    shutil.copy(project_dir / "datasets" / "line_sensors.parquet", bundle_dir / "bundle" / "line_sensors.parquet")