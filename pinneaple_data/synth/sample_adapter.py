"""PhysicalSample-like conversion and torchify utilities for synth module outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable, Union

import torch


@runtime_checkable
class PhysicalSampleLike(Protocol):
    """Protocol defining the minimal interface of a PhysicalSample-like object.

    Any object conforming to this protocol must expose:
      - `fields`: mapping of field names to data (often tensors/arrays)
      - `coords`: mapping of coordinate names to data (often tensors/arrays)
      - `meta`: mapping of metadata keys to arbitrary values

    This is used for duck-typing compatibility between the synth module and
    external `PhysicalSample` implementations.
    """
    fields: Dict[str, Any]
    coords: Dict[str, Any]
    meta: Dict[str, Any]


def has_pinnego_physical_sample() -> bool:
    """Check whether `pinneaple_data.physical_sample.PhysicalSample` is importable.

    Returns:
        True if `PhysicalSample` can be imported from `pinneaple_data`,
        otherwise False.
    """
    try:
        from pinneaple_data.physical_sample import PhysicalSample  # noqa: F401
        return True
    except Exception:
        return False


def _torchify_tree(obj: Any, *, device=None, dtype=None) -> Any:
    """
    Convert numpy/lists -> torch.Tensor recursively when possible.
    Keeps non-tensor metadata as is.

    Args:
        obj: Arbitrary nested structure (tensor, list/tuple, dict, numpy array, etc.).
        device: Optional device to move created/existing tensors to.
        dtype: Optional dtype to cast created/existing tensors to.

    Returns:
        A structure with the same overall shape as `obj`, where numeric arrays/lists
        are converted to `torch.Tensor` when feasible. Non-numeric / non-convertible
        objects are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        t = obj
        if device is not None:
            t = t.to(device=device)
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t

    # try tensor conversion for arrays/lists of numeric
    if isinstance(obj, (list, tuple)):
        # keep lists of strings intact
        if len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
        try:
            return torch.tensor(obj, device=device, dtype=dtype)  # type: ignore[arg-type]
        except Exception:
            return [_torchify_tree(x, device=device, dtype=dtype) for x in obj]

    if isinstance(obj, dict):
        return {k: _torchify_tree(v, device=device, dtype=dtype) for k, v in obj.items()}

    # numpy arrays
    try:
        import numpy as np  # noqa
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(device=device, dtype=dtype)  # type: ignore[arg-type]
    except Exception:
        pass

    return obj


def to_physical_sample(
    sample_like: Any,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """
    Best-effort conversion:
      - if already a PhysicalSample -> return
      - if duck-typed {fields, coords, meta} -> convert
      - else if dict with keys -> interpret

    Returns:
      - pinneaple_data.physical_sample.PhysicalSample if available
      - else returns a minimal dataclass fallback (SynthPhysicalSample)

    Args:
        sample_like: Input object to convert. Supported forms include:
            - An existing `pinneaple_data.physical_sample.PhysicalSample` instance
              (returned as-is).
            - Any object with attributes `fields`, `coords`, and `meta`.
            - A dict in one of the supported layouts:
                * {"fields": ..., "coords": ..., "meta": ...}
                * {"state": ..., "coords": ..., "meta": ...}
                * A plain dict treated as fields directly (excluding "coords"/"meta").
        device: Optional device (string or torch.device) to move tensors to.
        dtype: Optional torch dtype to cast tensors to.

    Returns:
        A `PhysicalSample` instance if the external implementation is available;
        otherwise, a locally-defined minimal dataclass `SynthPhysicalSample`
        carrying `fields`, `coords`, and `meta`.

    Raises:
        TypeError: If `sample_like` cannot be interpreted as a PhysicalSample-like
            object or dict.
    """
    device = torch.device(device) if device is not None and not isinstance(device, torch.device) else device

    # Already a PhysicalSample?
    if has_pinnego_physical_sample():
        from pinneaple_data.physical_sample import PhysicalSample  # type: ignore
        if isinstance(sample_like, PhysicalSample):
            return sample_like

    # Extract fields/coords/meta
    fields = None
    coords = None
    meta = None

    if hasattr(sample_like, "fields") and hasattr(sample_like, "coords") and hasattr(sample_like, "meta"):
        fields = getattr(sample_like, "fields")
        coords = getattr(sample_like, "coords")
        meta = getattr(sample_like, "meta")
    elif isinstance(sample_like, dict):
        # common patterns
        if "fields" in sample_like:
            fields = sample_like.get("fields")
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
        elif "state" in sample_like:
            fields = sample_like.get("state")
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
        else:
            # interpret dict as fields directly
            fields = {k: v for k, v in sample_like.items() if k not in ("coords", "meta")}
            coords = sample_like.get("coords", {})
            meta = sample_like.get("meta", {})
    else:
        raise TypeError("Cannot convert object to PhysicalSample: expected {fields,coords,meta} or dict-like.")

    fields_t = _torchify_tree(fields, device=device, dtype=dtype)
    coords_t = _torchify_tree(coords, device=device, dtype=dtype)
    meta = dict(meta) if isinstance(meta, dict) else {"meta": meta}

    # Preferred: real PhysicalSample
    if has_pinnego_physical_sample():
        from pinneaple_data.physical_sample import PhysicalSample  # type: ignore
        # Many projects define PhysicalSample differently; use the most conservative constructor:
        try:
            return PhysicalSample(fields=fields_t, coords=coords_t, meta=meta)
        except TypeError:
            # fallback constructor names
            try:
                return PhysicalSample(state=fields_t, coords=coords_t, meta=meta)
            except TypeError:
                # last resort: create and set attrs
                ps = PhysicalSample()
                setattr(ps, "fields", fields_t)
                setattr(ps, "coords", coords_t)
                setattr(ps, "meta", meta)
                return ps

    # Fallback minimal
    @dataclass
    class SynthPhysicalSample:
        """Minimal fallback PhysicalSample implementation for standalone usage.

        Attributes:
            fields: Field tensor mapping (typically containing "u", etc.).
            coords: Coordinate mapping (e.g., time and space grids).
            meta: Metadata mapping describing how the sample was generated.
        """
        fields: Dict[str, Any]
        coords: Dict[str, Any]
        meta: Dict[str, Any]

    return SynthPhysicalSample(fields=fields_t, coords=coords_t, meta=meta)