from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from pinneaple_arena.bundle.schema import BundleSchema


@dataclass(frozen=True)
class BundleData:
    root: Path
    manifest: Dict[str, Any]
    conditions: Dict[str, Any]
    points_collocation: pd.DataFrame
    points_boundary: pd.DataFrame
    sensors: Optional[pd.DataFrame] = None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(
    bundle_root: str | Path,
    *,
    schema: BundleSchema,
    require_sensors: bool = False,
) -> BundleData:
    root = Path(bundle_root)
    schema.validate_bundle_root(root)

    manifest = _read_json(root / "bundle" / "manifest.json")
    conditions = _read_json(root / "bundle" / "conditions.json")

    # validate manifest keys
    missing_keys = [k for k in schema.manifest_required_keys if k not in manifest]
    if missing_keys:
        raise RuntimeError(f"manifest.json missing keys: {missing_keys}")

    # validate regions exist in conditions.json
    # conditions.json includes meta keys too, but must contain the region keys
    for reg in schema.required_regions:
        if reg not in conditions:
            raise RuntimeError(f"conditions.json missing region '{reg}'")

    points_c = pd.read_parquet(root / "derived" / "points_collocation.parquet")
    points_b = pd.read_parquet(root / "derived" / "points_boundary.parquet")

    for col in ("x", "y"):
        if col not in points_c.columns:
            raise RuntimeError(f"points_collocation.parquet missing column '{col}'")
    for col in ("x", "y", "region"):
        if col not in points_b.columns:
            raise RuntimeError(f"points_boundary.parquet missing column '{col}'")

    regions = set(points_b["region"].astype(str).unique().tolist())
    missing = sorted(list(schema.required_regions - regions))
    if missing:
        raise RuntimeError(f"points_boundary.parquet missing regions: {missing}")

    sensors = None
    sensors_path = root / "bundle" / "sensors.parquet"
    if sensors_path.exists():
        sensors = pd.read_parquet(sensors_path)
        for c in schema.sensors_required_columns:
            if c not in sensors.columns:
                raise RuntimeError(f"sensors.parquet missing required column '{c}'")

    if require_sensors and sensors is None:
        raise RuntimeError(
            "Task requires sensors.parquet but it was not found.\n"
            f"Expected: {sensors_path}\n"
            "Fix:\n"
            " - export sensors.parquet from simulation (or measurement)\n"
            " - or set require_sensors=false in task config\n"
        )

    return BundleData(
        root=root,
        manifest=manifest,
        conditions=conditions,
        points_collocation=points_c,
        points_boundary=points_b,
        sensors=sensors,
    )
