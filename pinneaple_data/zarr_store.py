"""Zarr-backed UPD store for PhysicalSample persistence with columnar layout."""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import zarr

from .physical_sample import PhysicalSample


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _is_torch(x: Any) -> bool:
    """
    Check if the input is a torch.Tensor.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if x is a torch.Tensor and torch is available; otherwise False.
    """
    return torch is not None and isinstance(x, torch.Tensor)


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert array-like input to numpy.ndarray.

    Parameters
    ----------
    x : ArrayLike
        Input (numpy array or torch.Tensor).

    Returns
    -------
    np.ndarray
        NumPy array representation.
    """
    if _is_torch(x):
        x = x.detach().cpu()
        return np.asarray(x.numpy())
    return np.asarray(x)


def _rmtree_force(path: str, retries: int = 8, sleep_s: float = 0.25) -> None:
    """
    Robust directory removal for Windows (OneDrive/indexer/AV file locks).
    """
    if not os.path.exists(path):
        return

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
        except Exception:
            pass
        func(p)

    last_exc: Optional[BaseException] = None
    for _ in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError as e:
            last_exc = e
            time.sleep(sleep_s)
        except OSError as e:
            last_exc = e
            time.sleep(sleep_s)

    # final attempt
    if last_exc is not None:
        shutil.rmtree(path, onerror=_onerror)


def _atomic_replace_dir(tmp_dir: str, final_dir: str) -> None:
    """
    Atomically replace a directory when possible.

    On Windows, os.replace works for directories too, but can fail if target is open.
    We fallback to delete+rename.
    """
    if os.path.exists(final_dir):
        _rmtree_force(final_dir)
    os.replace(tmp_dir, final_dir)


def _create_array(group: Any, name: str, data: np.ndarray, chunks: Optional[Tuple[int, ...]] = None) -> Any:
    """
    Zarr v2/v3 compatibility wrapper.
    IMPORTANT: do NOT pass chunks=None to Zarr v3.
    """
    kwargs = dict(
        data=data,
        shape=data.shape,
        dtype=data.dtype,
        overwrite=True,
    )
    if chunks is not None:
        kwargs["chunks"] = chunks  # must be int or tuple[int,...]

    # Prefer create_dataset (v2-compatible)
    if hasattr(group, "create_dataset"):
        return group.create_dataset(name, **kwargs)

    # Zarr v3 create_array
    if hasattr(group, "create_array"):
        # create_array signature uses 'name=' explicitly in some versions
        return group.create_array(name=name, **kwargs)

    # Fallback
    if hasattr(group, "array"):
        arr_kwargs = dict(data=data, overwrite=True)
        if chunks is not None:
            arr_kwargs["chunks"] = chunks
        return group.array(name=name, **arr_kwargs)

    raise RuntimeError("Unsupported Zarr Group API: cannot create array")


class UPDZarrStore:
    """
    Minimal UPD Zarr store:

    root/
      fields/
        <field_name> : array with shape (N, ...)  (N = number of samples)
      coords/  (optional)
      meta/
        manifest.json (optional)
        info attrs

    This store assumes a "columnar" layout: each field is stacked across samples.
    """

    def __init__(self, root: str, mode: str = "r"):
        """
        Initialize the UPD Zarr store.

        Parameters
        ----------
        root : str
            Root directory path of the Zarr store.
        mode : str, optional
            Access mode ("r" for read-only, "w" for write). Default is "r".
        """
        self.root = str(root)
        self.mode = str(mode)

        # IMPORTANT:
        # - do not create groups in read-only mode
        self.grp = zarr.open_group(self.root, mode=self.mode)

        if self.mode.startswith("r"):
            # read-only: never create anything
            try:
                self.g_fields = self.grp["fields"]
            except Exception as e:
                raise FileNotFoundError(
                    f"UPDZarrStore: missing required group 'fields' in store: {self.root}. "
                    "This usually means the store was not written successfully or is incomplete."
                ) from e
            self.g_coords = self.grp.get("coords", None)
            self.g_meta = self.grp.get("meta", None)
        else:
            self.g_fields = self.grp.require_group("fields")
            self.g_coords = self.grp.require_group("coords")
            self.g_meta = self.grp.require_group("meta")

    @staticmethod
    def write(
        root: str,
        samples: Sequence[PhysicalSample],
        manifest: Optional[Dict[str, Any]] = None,
        *,
        overwrite: bool = True,
        chunks: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> str:
        """
        Write samples to a Zarr directory store.

        - Writes to <root>.tmp then atomically replaces <root>.
        - Creates required group 'fields' and stacks each field across samples.
        """
        root = str(root)
        tmp_root = root + ".tmp"

        if os.path.exists(tmp_root):
            _rmtree_force(tmp_root)

        if os.path.exists(root):
            if not overwrite:
                raise FileExistsError(f"Zarr store already exists: {root}")
            # DO NOT rely on zarr.open_group(mode='w') deletion behavior (Windows locks).
            # We'll delete ourselves later (atomic replace).
        os.makedirs(os.path.dirname(root) or ".", exist_ok=True)

        # Validate and collect available state keys
        if not samples:
            raise ValueError("UPDZarrStore.write: samples is empty")

        # Gather union of state keys (dict-state only for this writer)
        # For xr.Dataset, you should convert to dict or implement a separate writer.
        all_keys: List[str] = []
        for s in samples:
            if not isinstance(s.state, dict):
                raise TypeError(
                    "UPDZarrStore.write MVP expects PhysicalSample.state as dict[str, array/tensor]. "
                    "For xarray.Dataset, convert to dict or extend writer."
                )
            for k in s.state.keys():
                if k not in all_keys:
                    all_keys.append(k)

        # Stack per field
        stacked: Dict[str, np.ndarray] = {}
        for k in all_keys:
            arrs = []
            for i, s in enumerate(samples):
                if k not in s.state:
                    raise KeyError(f"Sample {i} missing field '{k}'. All samples must have same fields.")
                arrs.append(_to_numpy(s.state[k]))
            # ensure same shapes
            base_shape = arrs[0].shape
            for j, a in enumerate(arrs[1:], start=1):
                if a.shape != base_shape:
                    raise ValueError(
                        f"Field '{k}' has inconsistent shapes across samples: "
                        f"{base_shape} vs {a.shape} at sample {j}"
                    )
            stacked[k] = np.stack(arrs, axis=0)  # (N, ...)

        # Write tmp store
        grp = zarr.open_group(tmp_root, mode="w")
        g_fields = grp.require_group("fields")
        g_coords = grp.require_group("coords")
        g_meta = grp.require_group("meta")

        # store basic metadata
        grp.attrs["format"] = "pinneaple.upd.zarr"
        grp.attrs["version"] = "0.1"
        grp.attrs["num_samples"] = int(len(samples))
        grp.attrs["fields"] = list(all_keys)

        # provenance per-sample (optional)
        # Keep it light: store provenance as JSON lines in meta if present.
        prov_lines: List[str] = []
        for s in samples:
            try:
                prov_lines.append(json.dumps(s.provenance or {}, ensure_ascii=False))
            except Exception:
                prov_lines.append("{}")
        g_meta.attrs["provenance_jsonl"] = "\n".join(prov_lines)

        if manifest is not None:
            g_meta.attrs["manifest_json"] = json.dumps(manifest, ensure_ascii=False)

        # Write arrays
        chunks = chunks or {}
        for k, data in stacked.items():
            ck = chunks.get(k, None)
            _create_array(g_fields, k, data, chunks=ck)

        # (coords) left empty for dict-state MVP, but group exists for consistency
        g_coords.attrs["note"] = "coords group reserved (grid coords / mesh coords / time axes)"

        # Atomically replace final
        _atomic_replace_dir(tmp_root, root)
        return root

    def num_samples(self) -> int:
        """
        Return the number of samples in the store.

        Returns
        -------
        int
            Sample count.
        """
        n = self.grp.attrs.get("num_samples", None)
        if n is not None:
            return int(n)
        # fallback: infer from first field
        keys = list(self.g_fields.array_keys()) if hasattr(self.g_fields, "array_keys") else list(self.g_fields.keys())
        if not keys:
            return 0
        a0 = self.g_fields[keys[0]]
        return int(a0.shape[0])

    def field_names(self) -> List[str]:
        """
        Return the list of field names stored in the store.

        Returns
        -------
        List[str]
            Field names.
        """
        keys = self.grp.attrs.get("fields", None)
        if keys is not None:
            return list(keys)
        # fallback
        if hasattr(self.g_fields, "array_keys"):
            return list(self.g_fields.array_keys())
        return list(self.g_fields.keys())

    def read_sample(
        self,
        idx: int,
        *,
        fields: Optional[Sequence[str]] = None,
    ) -> PhysicalSample:
        """
        Read one sample from stacked field arrays.
        """
        idx = int(idx)
        names = list(fields) if fields is not None else self.field_names()
        state: Dict[str, Any] = {}
        for k in names:
            if k not in self.g_fields:
                raise KeyError(f"Missing field '{k}' in store: {self.root}")
            arr = self.g_fields[k]
            x = np.asarray(arr[idx])
            # return torch tensors if torch available
            if torch is not None:
                state[k] = torch.from_numpy(x)
            else:
                state[k] = x

        provenance: Dict[str, Any] = {"idx": idx}
        # Try to recover provenance line if available
        try:
            meta = self.grp.get("meta", None)
            if meta is not None:
                prov_jsonl = meta.attrs.get("provenance_jsonl", "")
                if prov_jsonl:
                    lines = prov_jsonl.splitlines()
                    if 0 <= idx < len(lines):
                        provenance.update(json.loads(lines[idx]))
        except Exception:
            pass

        return PhysicalSample(
            state=state,
            domain={"type": "grid"},
            provenance=provenance,
        )
