"""Serialization utilities for PhysicalSample to/from PT, Zarr, and HDF5 formats."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import json

import torch

from .physical_sample import PhysicalSample


def save_pt(samples: Sequence[PhysicalSample], path: str) -> None:
    """
    Save a sequence of PhysicalSamples to a PyTorch (.pt) file.

    Parameters
    ----------
    samples : Sequence[PhysicalSample]
        Sequence of PhysicalSample objects to serialize.
    path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = []
    for s in samples:
        payload.append(
            {
                "fields": s.fields,
                "coords": s.coords,
                "meta": s.meta,
            }
        )
    torch.save(payload, path)


def load_pt(path: str) -> List[PhysicalSample]:
    """
    Load PhysicalSamples from a PyTorch (.pt) file.

    Parameters
    ----------
    path : str
        Path to the .pt file.

    Returns
    -------
    List[PhysicalSample]
        List of deserialized PhysicalSample objects.
    """
    payload = torch.load(path, map_location="cpu")
    out: List[PhysicalSample] = []
    for item in payload:
        out.append(PhysicalSample(fields=item["fields"], coords=item.get("coords", {}), meta=item.get("meta", {})))
    return out


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    """
    Save a manifest dictionary to a JSON file.

    Parameters
    ----------
    path : str
        Output file path.
    manifest : Dict[str, Any]
        Manifest dictionary to serialize.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> Dict[str, Any]:
    """
    Load a manifest dictionary from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    Dict[str, Any]
        Deserialized manifest dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_zarr(samples: Sequence[PhysicalSample], root: str, *, compressor: str = "default") -> None:
    """
    Save PhysicalSamples to a Zarr directory store (legacy implementation).

    Parameters
    ----------
    samples : Sequence[PhysicalSample]
        Sequence of PhysicalSample objects to serialize.
    root : str
        Root directory path for the Zarr store.
    compressor : str, optional
        Compression method. Default is "default".
    """
    import zarr  # optional
    import numpy as np

    os.makedirs(root, exist_ok=True)
    grp = zarr.open_group(root, mode="w")

    # store meta as json strings per sample (simple, robust)
    metas = [json.dumps(s.meta) for s in samples]
    grp.create_dataset("meta_json", data=np.array(metas, dtype=object), dtype=object)

    # store fields/coords as separate groups
    g_fields = grp.create_group("fields")
    g_coords = grp.create_group("coords")

    # naive approach: all samples must have the same field keys + shapes
    # industry-friendly: deterministic structure
    field_keys = sorted(samples[0].fields.keys())
    coord_keys = sorted(samples[0].coords.keys())

    grp.attrs["field_keys"] = field_keys
    grp.attrs["coord_keys"] = coord_keys
    grp.attrs["count"] = len(samples)

    for k in field_keys:
        arr0 = samples[0].fields[k]
        if not torch.is_tensor(arr0):
            # store non-tensor as meta only
            continue
        data = torch.stack([s.fields[k].cpu() for s in samples], dim=0).numpy()
        g_fields.create_dataset(k, data=data, chunks=(1, *data.shape[1:]))

    for k in coord_keys:
        arr0 = samples[0].coords[k]
        if not torch.is_tensor(arr0):
            continue
        data = torch.stack([s.coords[k].cpu() for s in samples], dim=0).numpy()
        g_coords.create_dataset(k, data=data, chunks=(1, *data.shape[1:]))


def load_zarr(root: str) -> List[PhysicalSample]:
    """
    Load PhysicalSamples from a Zarr directory store (legacy implementation).

    Parameters
    ----------
    root : str
        Root directory path of the Zarr store.

    Returns
    -------
    List[PhysicalSample]
        List of deserialized PhysicalSample objects.
    """
    import zarr  # optional
    import numpy as np

    grp = zarr.open_group(root, mode="r")
    n = int(grp.attrs["count"])
    field_keys = list(grp.attrs.get("field_keys", []))
    coord_keys = list(grp.attrs.get("coord_keys", []))

    metas = [json.loads(x) for x in grp["meta_json"][:]]

    out: List[PhysicalSample] = []
    for i in range(n):
        fields = {}
        coords = {}
        for k in field_keys:
            if k in grp["fields"]:
                fields[k] = torch.from_numpy(grp["fields"][k][i])
        for k in coord_keys:
            if k in grp["coords"]:
                coords[k] = torch.from_numpy(grp["coords"][k][i])
        out.append(PhysicalSample(fields=fields, coords=coords, meta=metas[i]))
    return out


def save_hdf5(samples: Sequence[PhysicalSample], path: str) -> None:
    """
    Save PhysicalSamples to an HDF5 file.

    Parameters
    ----------
    samples : Sequence[PhysicalSample]
        Sequence of PhysicalSample objects to serialize.
    path : str
        Output HDF5 file path.
    """
    import h5py  # optional
    import numpy as np

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["count"] = len(samples)
        meta_json = [json.dumps(s.meta) for s in samples]
        f.create_dataset("meta_json", data=np.array(meta_json, dtype=h5py.string_dtype()))

        g_fields = f.create_group("fields")
        g_coords = f.create_group("coords")

        field_keys = sorted(samples[0].fields.keys())
        coord_keys = sorted(samples[0].coords.keys())

        f.attrs["field_keys"] = json.dumps(field_keys)
        f.attrs["coord_keys"] = json.dumps(coord_keys)

        for k in field_keys:
            v0 = samples[0].fields[k]
            if not torch.is_tensor(v0):
                continue
            data = torch.stack([s.fields[k].cpu() for s in samples], dim=0).numpy()
            g_fields.create_dataset(k, data=data, compression="gzip", chunks=True)

        for k in coord_keys:
            v0 = samples[0].coords[k]
            if not torch.is_tensor(v0):
                continue
            data = torch.stack([s.coords[k].cpu() for s in samples], dim=0).numpy()
            g_coords.create_dataset(k, data=data, compression="gzip", chunks=True)


def load_hdf5(path: str) -> List[PhysicalSample]:
    """
    Load PhysicalSamples from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.

    Returns
    -------
    List[PhysicalSample]
        List of deserialized PhysicalSample objects.
    """
    import h5py  # optional
    with h5py.File(path, "r") as f:
        n = int(f.attrs["count"])
        metas = [json.loads(s) for s in f["meta_json"][:]]
        field_keys = json.loads(f.attrs["field_keys"])
        coord_keys = json.loads(f.attrs["coord_keys"])

        out: List[PhysicalSample] = []
        for i in range(n):
            fields, coords = {}, {}
            for k in field_keys:
                if k in f["fields"]:
                    fields[k] = torch.from_numpy(f["fields"][k][i])
            for k in coord_keys:
                if k in f["coords"]:
                    coords[k] = torch.from_numpy(f["coords"][k][i])
            out.append(PhysicalSample(fields=fields, coords=coords, meta=metas[i]))
        return out


def save_zarr(samples, root: str, *, compressor: str = "default") -> None:
    """
    Save PhysicalSamples to a Zarr directory store using UPDZarrStore.

    Parameters
    ----------
    samples : Sequence
        Sequence of PhysicalSample-like objects to serialize.
    root : str
        Root directory path for the Zarr store.
    compressor : str, optional
        Compression method. Default is "default".
    """
    from .zarr_store import UPDZarrStore, ZarrWriteSpec
    UPDZarrStore.write(root, samples, manifest=None, spec=ZarrWriteSpec(chunk_by_sample=True))

def load_zarr(root: str):
    """
    Load PhysicalSamples from a Zarr directory store using UPDZarrStore.

    Parameters
    ----------
    root : str
        Root directory path of the Zarr store.

    Returns
    -------
    list
        List of PhysicalSample-like objects from the store.
    """
    from .zarr_store import UPDZarrStore
    store = UPDZarrStore(root, mode="r")
    return list(store.iter_samples())
