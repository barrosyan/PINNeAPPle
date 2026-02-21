"""UPD shard loading and PhysicalSample conversion adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import xarray as xr


# Accept:
#  - tuple(zarr_path, meta_path)
#  - dict with {"zarr_path":..., "meta_path":...}
#  - already-opened xarray.Dataset + meta dict
#  - UPDItem (from pinneaple_pinn.io) with .open_dataset() / .load_meta()
UPDInput = Union[
    Tuple[str, str],
    Dict[str, Any],
    Any,
]


def load_upd_item(upd: UPDInput) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Load a UPD shard dataset and its associated metadata.

    Parameters
    ----------
    upd : UPDInput
        Input describing the UPD shard. Supported forms:
        - Tuple[str, str] representing (zarr_path, meta_path)
        - Dict with {"zarr_path": ..., "meta_path": ...}
        - Dict with {"ds": xarray.Dataset, "meta": dict}
        - UPDItem-like object exposing open_dataset() and load_meta()

    Returns
    -------
    Tuple[xr.Dataset, Dict[str, Any]]
        Loaded xarray Dataset and metadata dictionary.

    Raises
    ------
    TypeError
        If the input type or structure is unsupported.
    """
    """
    Load UPD shard data (Zarr) and metadata (JSON).

    Supports:
      - (zarr_path, meta_path)
      - {"zarr_path":..., "meta_path":...}
      - UPDItem-like object (has open_dataset/load_meta)
      - (Dataset, meta_dict) packed in dict: {"ds":..., "meta":...}
    """
    # UPDItem-like
    if hasattr(upd, "open_dataset") and hasattr(upd, "load_meta"):
        ds = upd.open_dataset()
        meta = upd.load_meta()
        return ds, meta

    # dict forms
    if isinstance(upd, dict):
        if "ds" in upd and "meta" in upd:
            ds = upd["ds"]
            meta = upd["meta"]
            if not isinstance(ds, xr.Dataset):
                raise TypeError("upd['ds'] must be an xarray.Dataset")
            if not isinstance(meta, dict):
                raise TypeError("upd['meta'] must be a dict")
            return ds, meta

        zarr_path = upd.get("zarr_path")
        meta_path = upd.get("meta_path")
        if zarr_path and meta_path:
            return _load_paths(zarr_path, meta_path)

    # tuple paths
    if isinstance(upd, (tuple, list)) and len(upd) == 2:
        return _load_paths(upd[0], upd[1])

    raise TypeError(f"Unsupported UPD input type: {type(upd)}")


def _load_paths(zarr_path: str, meta_path: str) -> Tuple[xr.Dataset, Dict[str, Any]]:
    """
    Load a Zarr dataset and corresponding JSON metadata from filesystem paths.

    Parameters
    ----------
    zarr_path : str
        Filesystem path to the Zarr store.
    meta_path : str
        Filesystem path to the metadata JSON file.

    Returns
    -------
    Tuple[xr.Dataset, Dict[str, Any]]
        Loaded dataset and parsed metadata dictionary.
    """
    ds = xr.open_zarr(str(zarr_path))
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return ds, meta


def _infer_domain_from_upd(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Infer domain metadata from a UPD xarray Dataset.

    This function inspects coordinates and dimension sizes to determine
    grid structure and coordinate system assumptions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset loaded from a UPD shard.

    Returns
    -------
    Dict[str, Any]
        Domain dictionary describing grid type, coordinate system,
        and dimensionality information.
    """
    """
    Domain inference for UPD grid shards.

    Typical coords:
      - time, lat, lon, (optional lev)
    """
    coords = set(ds.coords)
    dims = dict(ds.sizes)

    domain: Dict[str, Any] = {"type": "grid"}

    # common cases
    if {"lat", "lon"}.issubset(coords):
        domain["grid"] = {
            "coord_system": "latlon",
            "has_time": "time" in coords,
            "has_lev": "lev" in coords,
            "dims": {k: int(v) for k, v in dims.items()},
        }
    else:
        domain["grid"] = {
            "coord_system": "unknown",
            "dims": {k: int(v) for k, v in dims.items()},
        }
    return domain


def upd_to_physical_sample(
    upd: UPDInput,
    *,
    schema_override: Optional[Dict[str, Any]] = None,
    provenance_extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convert a UPD shard into a PhysicalSample instance.

    Parameters
    ----------
    upd : UPDInput
        UPD shard input (see load_upd_item).
    schema_override : Optional[Dict[str, Any]]
        Optional schema dictionary overriding metadata schema.
    provenance_extra : Optional[Dict[str, Any]]
        Additional provenance entries to merge into the generated metadata.

    Returns
    -------
    Any
        Instance of PhysicalSample containing dataset, schema,
        domain, and provenance information.

    Raises
    ------
    ImportError
        If PhysicalSample class cannot be imported.
    """
    """
    Creates a PhysicalSample from a UPD shard (Zarr+JSON).

    Requires `pinneaple_data.physical_sample.PhysicalSample` to exist.

    Returns:
      PhysicalSample

    If you haven't created PhysicalSample yet, this will raise ImportError.
    """
    ds, meta = load_upd_item(upd)
    schema = schema_override or meta.get("schema") or {}
    domain = _infer_domain_from_upd(ds)

    provenance = {
        "source": "upd",
        "uid": meta.get("uid"),
        "zarr_path": meta.get("zarr_path") or meta.get("paths", {}).get("zarr"),
        "meta_path": meta.get("meta_path"),
        "hub_query": meta.get("hub_query"),
        "spacetime": meta.get("spacetime"),
        "selection": meta.get("selection"),
        "shard": meta.get("shard"),
    }
    if provenance_extra:
        provenance.update(provenance_extra)

    try:
        from pinneaple_data.physical_sample import PhysicalSample  # type: ignore
    except Exception as e:
        raise ImportError(
            "PhysicalSample not found. Create pinneaple_data/physical_sample.py with PhysicalSample dataclass."
        ) from e

    sample = PhysicalSample(
        state=ds,
        geometry=None,
        schema=schema,
        domain=domain,
        provenance=provenance,
    )
    return sample


def attach_upd_state(sample: Any, upd: UPDInput) -> Any:
    """
    Attach UPD dataset state and metadata to an existing sample object.

    This function updates state, schema, domain, and provenance fields.
    It supports both dictionary-based and attribute-based sample structures.

    Parameters
    ----------
    sample : Any
        Existing PhysicalSample-like object or dictionary.
    upd : UPDInput
        UPD shard input to load and attach.

    Returns
    -------
    Any
        Updated sample with attached dataset and metadata.
    """
    """
    Attaches UPD state (xarray.Dataset) and schema/domain/provenance into an existing sample.

    Works with:
      - PhysicalSample-like object (attributes)
      - dict-like sample
    """
    ds, meta = load_upd_item(upd)
    schema = meta.get("schema") or {}
    domain = _infer_domain_from_upd(ds)

    provenance = {
        "source": "upd",
        "uid": meta.get("uid"),
        "hub_query": meta.get("hub_query"),
        "spacetime": meta.get("spacetime"),
        "selection": meta.get("selection"),
        "shard": meta.get("shard"),
    }

    if isinstance(sample, dict):
        sample["state"] = ds
        sample["schema"] = schema
        sample["domain"] = domain
        prov = sample.get("provenance", {}) or {}
        if isinstance(prov, dict):
            prov.update(provenance)
        else:
            prov = provenance
        sample["provenance"] = prov
        return sample

    if hasattr(sample, "state"):
        setattr(sample, "state", ds)
    if hasattr(sample, "schema"):
        setattr(sample, "schema", schema)
    if hasattr(sample, "domain"):
        setattr(sample, "domain", domain)
    if hasattr(sample, "provenance"):
        prov = getattr(sample, "provenance") or {}
        if isinstance(prov, dict):
            prov.update(provenance)
        else:
            prov = provenance
        setattr(sample, "provenance", prov)

    return sample