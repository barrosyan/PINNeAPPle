from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pinneaple_arena.io.yamlx import load_yaml


@dataclass(frozen=True)
class BundleSchema:
    """
    Schema definition for validating a PINNeAPPle bundle.

    Attributes
    ----------
    schema_version : str
        Version identifier of the bundle schema.
    required_files : List[str]
        Relative file paths that must exist under the bundle root.
    required_regions : Set[str]
        Set of region names that must appear in conditions.json and boundary data.
    manifest_required_keys : Set[str]
        Required keys that must exist in manifest.json.
    sensors_required_columns : List[str]
        Required column names for sensors.parquet if present.
    """

    schema_version: str
    required_files: List[str]
    required_regions: Set[str]
    manifest_required_keys: Set[str]
    sensors_required_columns: List[str]

    def validate_bundle_root(self, bundle_root: str | Path) -> None:
        """
        Validate that the bundle root directory exists and contains
        all required files defined in the schema.

        Parameters
        ----------
        bundle_root : str | Path
            Root directory of the bundle.

        Raises
        ------
        FileNotFoundError
            If the bundle root directory does not exist.
        RuntimeError
            If any required files are missing.
        """
        root = Path(bundle_root)
        if not root.exists():
            raise FileNotFoundError(f"Bundle root not found: {root}")

        # required files
        missing = []
        for rel in self.required_files:
            if not (root / rel).exists():
                missing.append(rel)
        if missing:
            raise RuntimeError(
                "Bundle is missing required files:\n"
                + "\n".join(f" - {m}" for m in missing)
                + "\n\nFix:\n"
                " - generate these files under the bundle_root\n"
                " - or update configs/data/bundle_schema.yaml if you intentionally changed geometry file type (usd vs stl)\n"
            )


def load_bundle_schema(path: str | Path) -> BundleSchema:
    """
    Load a bundle schema definition from a YAML configuration file.

    The YAML file must define:
        - required_files (non-empty list)
        - optional condition, manifest, and sensor validation rules

    Parameters
    ----------
    path : str | Path
        Path to the bundle_schema.yaml file.

    Returns
    -------
    BundleSchema
        Parsed and validated schema object.

    Raises
    ------
    RuntimeError
        If required fields in the YAML configuration are missing or invalid.
    """
    cfg = load_yaml(path)

    ver = str(cfg.get("bundle_schema_version", "1.0"))

    required_files = cfg.get("required_files", [])
    if not isinstance(required_files, list) or not required_files:
        raise RuntimeError("bundle_schema.yaml must define non-empty required_files list")

    cond = cfg.get("conditions_json", {})
    required_regions = set(cond.get("required_regions", ["inlet", "outlet", "walls", "obstacle"]))

    man = cfg.get("manifest_json", {})
    man_keys = set(man.get("required_keys", ["problem_id", "pde", "nu", "domain", "fields", "weights"]))

    sensors = cfg.get("sensors_parquet", {})
    sensors_required_cols = sensors.get("required_columns", ["x", "y", "u", "v", "scenario_id", "split"])
    if not isinstance(sensors_required_cols, list):
        raise RuntimeError("sensors_parquet.required_columns must be a list")

    return BundleSchema(
        schema_version=ver,
        required_files=[str(x) for x in required_files],
        required_regions=required_regions,
        manifest_required_keys=man_keys,
        sensors_required_columns=[str(x) for x in sensors_required_cols],
    )