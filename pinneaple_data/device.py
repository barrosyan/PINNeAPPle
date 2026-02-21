"""Device utilities for pinning and moving PhysicalSample tensors between CPU and GPU."""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch


def _pin(x: Any) -> Any:
    """
    Attempt to pin a tensor to CPU pinned memory.

    If the input is a CPU tensor, this function tries to call
    `pin_memory()` to enable faster asynchronous GPU transfers.
    If pinning fails or the input is not a CPU tensor, the original
    object is returned unchanged.

    Parameters
    ----------
    x : Any
        Object that may be a torch.Tensor.

    Returns
    -------
    Any
        Pinned tensor if applicable, otherwise the original object.
    """
    if torch.is_tensor(x) and x.device.type == "cpu":
        try:
            return x.pin_memory()
        except Exception:
            return x
    return x


def _to(x: Any, device: torch.device, dtype: Optional[torch.dtype], non_blocking: bool) -> Any:
    """
    Move a tensor to a target device and optionally cast dtype.

    If the input is a torch.Tensor, it will be:
      - Cast to the provided dtype (if specified and different),
      - Transferred to the target device.

    Non-tensor inputs are returned unchanged.

    Parameters
    ----------
    x : Any
        Object that may be a torch.Tensor.
    device : torch.device
        Target device.
    dtype : Optional[torch.dtype]
        Target data type (optional).
    non_blocking : bool
        Whether the device transfer should be asynchronous when possible.

    Returns
    -------
    Any
        Transformed tensor if applicable, otherwise the original object.
    """
    if torch.is_tensor(x):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        return x.to(device=device, non_blocking=non_blocking)
    return x


def pin_sample(sample: Any) -> Any:
    """
    Pin all tensor entries inside a PhysicalSample-like structure.

    This function supports two structural formats:
      - An object with `fields` and `coords` attributes.
      - A dictionary containing "fields" and "coords" keys.

    All tensors found inside `fields` and `coords` are passed
    through `_pin()`.

    Parameters
    ----------
    sample : Any
        PhysicalSample-like object or dictionary.

    Returns
    -------
    Any
        Modified sample with pinned tensors where applicable.
    """
    if hasattr(sample, "fields") and hasattr(sample, "coords"):
        sample.fields = {k: _pin(v) for k, v in sample.fields.items()}
        sample.coords = {k: _pin(v) for k, v in sample.coords.items()}
        return sample

    if isinstance(sample, dict) and "fields" in sample and "coords" in sample:
        sample["fields"] = {k: _pin(v) for k, v in sample["fields"].items()}
        sample["coords"] = {k: _pin(v) for k, v in sample["coords"].items()}
        return sample

    return sample


def to_device_sample(
    sample: Any,
    device: str | torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True
) -> Any:
    """
    Move all tensor entries inside a PhysicalSample-like structure
    to a target device.

    This function supports two structural formats:
      - An object with `fields` and `coords` attributes.
      - A dictionary containing "fields" and "coords" keys.

    All tensors found inside `fields` and `coords` are passed
    through `_to()`.

    Parameters
    ----------
    sample : Any
        PhysicalSample-like object or dictionary.
    device : str or torch.device
        Target device.
    dtype : Optional[torch.dtype], optional
        Target data type for tensors.
    non_blocking : bool, optional
        Whether transfers should be asynchronous when supported.

    Returns
    -------
    Any
        Modified sample with tensors moved to the target device.
    """
    dev = torch.device(device)

    if hasattr(sample, "fields") and hasattr(sample, "coords"):
        sample.fields = {k: _to(v, dev, dtype, non_blocking) for k, v in sample.fields.items()}
        sample.coords = {k: _to(v, dev, dtype, non_blocking) for k, v in sample.coords.items()}
        return sample

    if isinstance(sample, dict) and "fields" in sample and "coords" in sample:
        sample["fields"] = {k: _to(v, dev, dtype, non_blocking) for k, v in sample["fields"].items()}
        sample["coords"] = {k: _to(v, dev, dtype, non_blocking) for k, v in sample["coords"].items()}
        return sample

    return sample