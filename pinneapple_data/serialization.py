"""Serialization utilities for PhysicalSample to/from PT, Zarr, and HDF5 formats.

This module is backward/forward compatible:
- New UPD PhysicalSample: state/geometry/schema/domain/provenance/extras
- Legacy PhysicalSample: fields/coords/meta
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Optional
import os
import json

import torch

from .physical_sample import PhysicalSample


# -------------------------
# Manifest helpers
# -------------------------
def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Compatibility helpers
# -------------------------
def _is_new(sample: Any) -> bool:
    return hasattr(sample, "state")


def _to_payload(sample: PhysicalSample) -> Dict[str, Any]:
    """
    Convert PhysicalSample -> serializable payload.

    New format payload keys:
      state, geometry, schema, domain, provenance, extras, _format="upd_v1"
    Legacy format payload keys:
      fields, coords, meta, _format="legacy_v0"
    """
    if _is_new(sample):
        return {
            "_format": "upd_v1",
            "state": sample.state,
            "geometry": getattr(sample, "geometry", None),
            "schema": getattr(sample, "schema", {}) or {},
            "domain": getattr(sample, "domain", {}) or {},
            "provenance": getattr(sample, "provenance", {}) or {},
            "extras": getattr(sample, "extras", {}) or {},
        }

    # legacy fallback
    return {
        "_format": "legacy_v0",
        "fields": getattr(sample, "fields", {}) or {},
        "coords": getattr(sample, "coords", {}) or {},
        "meta": getattr(sample, "meta", {}) or {},
    }


def _from_payload(item: Dict[str, Any]) -> PhysicalSample:
    """
    Convert payload -> PhysicalSample (always returns the NEW PhysicalSample class).
    """
    fmt = item.get("_format")

    # New format (preferred)
    if fmt == "upd_v1" or ("state" in item):
        return PhysicalSample(
            state=item["state"],
            geometry=item.get("geometry"),
            schema=item.get("schema", {}) or {},
            domain=item.get("domain", {}) or {},
            provenance=item.get("provenance", {}) or {},
            extras=item.get("extras", {}) or {},
        )

    # Legacy -> map into new
    fields = item.get("fields", {}) or {}
    coords = item.get("coords", {}) or {}
    meta = item.get("meta", {}) or {}

    state: Dict[str, Any] = {}
    # Legacy tended to store coords separately; we merge into state
    if isinstance(coords, dict):
        state.update(coords)
    if isinstance(fields, dict):
        state.update(fields)

    schema: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}

    if isinstance(meta, dict):
        # common legacy conventions
        if "units" in meta:
            schema["units"] = meta["units"]
        if "sample_id" in meta:
            provenance["sample_id"] = meta["sample_id"]
        extras["legacy_meta"] = meta

    # domain type is unknown unless legacy meta encoded it
    domain = {"type": (meta or {}).get("domain_type", "unknown")} if isinstance(meta, dict) else {"type": "unknown"}

    return PhysicalSample(
        state=state,
        geometry=None,
        schema=schema,
        domain=domain,
        provenance=provenance,
        extras=extras,
    )


# -------------------------
# PT
# -------------------------
def save_pt(samples: Sequence[PhysicalSample], path: str) -> None:
    """
    Save a sequence of PhysicalSamples to a PyTorch (.pt) file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = [_to_payload(s) for s in samples]
    torch.save(payload, path)


def load_pt(path: str) -> List[PhysicalSample]:
    """
    Load PhysicalSamples from a PyTorch (.pt) file.
    """
    payload = torch.load(path, map_location="cpu")
    out: List[PhysicalSample] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("Invalid .pt payload: expected list[dict].")
        out.append(_from_payload(item))
    return out


# -------------------------
# HDF5
# -------------------------
def save_hdf5(samples: Sequence[PhysicalSample], path: str) -> None:
    """
    Save PhysicalSamples to an HDF5 file.

    Strategy:
      - store each sample as a torch-saved binary blob (portable, supports tensors)
      - store small manifest attrs
    """
    import h5py  # optional
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["count"] = len(samples)
        f.attrs["format"] = "upd_v1_or_legacy_v0"

        blobs = []
        for s in samples:
            obj = _to_payload(s)
            blobs.append(torch.save(obj, _BytesIO(), _use_new_zipfile_serialization=True))  # placeholder


# NOTE: We need a real BytesIO helper; keep below outside the HDF5 function to avoid import cycles.
class _BytesIO:
    """
    Minimal BytesIO-like object for torch.save into bytes without importing io in hot paths.
    """
    def __init__(self):
        import io
        self._b = io.BytesIO()
    def write(self, x):
        return self._b.write(x)
    def getvalue(self):
        return self._b.getvalue()
    def seek(self, pos, whence=0):
        return self._b.seek(pos, whence)


def save_hdf5(samples: Sequence[PhysicalSample], path: str) -> None:
    import h5py  # optional
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["count"] = len(samples)
        f.attrs["payload_kind"] = "torch_bytes"

        dt = h5py.vlen_dtype(np.dtype("uint8"))
        ds = f.create_dataset("payload_bytes", shape=(len(samples),), dtype=dt)

        for i, s in enumerate(samples):
            bio = _BytesIO()
            torch.save(_to_payload(s), bio._b)
            data = np.frombuffer(bio.getvalue(), dtype=np.uint8)
            ds[i] = data


def load_hdf5(path: str) -> List[PhysicalSample]:
    import h5py  # optional
    import numpy as np
    import io

    with h5py.File(path, "r") as f:
        n = int(f.attrs["count"])
        if "payload_bytes" in f:
            out: List[PhysicalSample] = []
            for i in range(n):
                raw = f["payload_bytes"][i]
                buf = io.BytesIO(np.array(raw, dtype=np.uint8).tobytes())
                item = torch.load(buf, map_location="cpu")
                if not isinstance(item, dict):
                    raise TypeError("Invalid HDF5 payload: expected dict.")
                out.append(_from_payload(item))
            return out

        # Legacy fallback (if someone saved with the old layout)
        # Attempt to reconstruct fields/coords/meta layout
        if "meta_json" in f and "fields" in f and "coords" in f:
            metas = [json.loads(s) for s in f["meta_json"][:]]
            field_keys = json.loads(f.attrs.get("field_keys", "[]"))
            coord_keys = json.loads(f.attrs.get("coord_keys", "[]"))

            out: List[PhysicalSample] = []
            for i in range(n):
                fields, coords = {}, {}
                for k in field_keys:
                    if k in f["fields"]:
                        fields[k] = torch.from_numpy(f["fields"][k][i])
                for k in coord_keys:
                    if k in f["coords"]:
                        coords[k] = torch.from_numpy(f["coords"][k][i])

                out.append(_from_payload({"_format": "legacy_v0", "fields": fields, "coords": coords, "meta": metas[i]}))
            return out

        raise ValueError("Unrecognized HDF5 layout.")


# -------------------------
# Zarr (official path)
# -------------------------
def save_zarr(samples, root: str, *, compressor: str = "default") -> None:
    """
    Save PhysicalSamples to a Zarr directory store using UPDZarrStore.
    """
    from .zarr_store import UPDZarrStore, ZarrWriteSpec
    UPDZarrStore.write(root, samples, manifest=None, spec=ZarrWriteSpec(chunk_by_sample=True))


def load_zarr(root: str):
    """
    Load PhysicalSamples from a Zarr directory store using UPDZarrStore.
    """
    from .zarr_store import UPDZarrStore
    store = UPDZarrStore(root, mode="r")
    return list(store.iter_samples())