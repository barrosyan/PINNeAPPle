"""Layer-freezing utilities for discriminative fine-tuning of PINNs.

All public functions operate on ``torch.nn.Module`` objects and match
parameter names using simple prefix matching against the strings returned
by ``model.named_parameters()``.

Example
-------
>>> frozen_count = freeze_layers(model, prefixes=["encoder", "trunk.0"])
>>> groups = layer_lr_groups(model, base_lr=1e-4, scale_dict={"encoder": 0.01})
>>> opt = torch.optim.Adam(groups)
"""
from __future__ import annotations

from typing import Dict, List

import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(name: str, prefixes: List[str]) -> bool:
    """Return True if *name* starts with any prefix in *prefixes*."""
    return any(name == p or name.startswith(p + ".") or name.startswith(p)
               for p in prefixes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def freeze_layers(model: nn.Module, prefixes: List[str]) -> int:
    """Freeze parameters whose names start with any of *prefixes*.

    Parameters
    ----------
    model:
        The ``nn.Module`` to modify in-place.
    prefixes:
        A list of parameter-name prefixes.  An empty list is a no-op.

    Returns
    -------
    int
        The number of individual parameter tensors that were frozen.

    Example
    -------
    >>> n = freeze_layers(model, ["net.0", "net.1"])
    >>> print(f"Froze {n} parameter tensors")
    """
    if not prefixes:
        return 0
    count = 0
    for name, param in model.named_parameters():
        if _matches_any(name, prefixes):
            param.requires_grad_(False)
            count += 1
    return count


def unfreeze_layers(model: nn.Module, prefixes: List[str]) -> int:
    """Unfreeze parameters whose names start with any of *prefixes*.

    Parameters
    ----------
    model:
        The ``nn.Module`` to modify in-place.
    prefixes:
        A list of parameter-name prefixes.  An empty list is a no-op.

    Returns
    -------
    int
        The number of individual parameter tensors that were unfrozen.
    """
    if not prefixes:
        return 0
    count = 0
    for name, param in model.named_parameters():
        if _matches_any(name, prefixes):
            param.requires_grad_(True)
            count += 1
    return count


def freeze_all_except(model: nn.Module, trainable_prefixes: List[str]) -> None:
    """Freeze every parameter that does *not* match *trainable_prefixes*.

    This is the canonical setup for the ``"feature_extract"`` strategy: the
    backbone is frozen and only the head (or adapter layers) are trained.

    Parameters
    ----------
    model:
        The ``nn.Module`` to modify in-place.
    trainable_prefixes:
        Parameters matching these prefixes keep ``requires_grad=True``.
        All others are frozen.  Pass an empty list to freeze everything.

    Example
    -------
    >>> freeze_all_except(model, trainable_prefixes=["head", "adapter"])
    """
    for name, param in model.named_parameters():
        if trainable_prefixes and _matches_any(name, trainable_prefixes):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def layer_lr_groups(
    model: nn.Module,
    base_lr: float,
    scale_dict: Dict[str, float],
) -> List[dict]:
    """Build optimizer param groups for discriminative / layerwise learning rates.

    Parameters that match a prefix in *scale_dict* receive
    ``lr = base_lr * scale``.  All remaining *trainable* parameters receive
    ``lr = base_lr``.  Frozen (``requires_grad=False``) parameters are
    excluded from every group.

    Parameters
    ----------
    model:
        Source of ``named_parameters()``.
    base_lr:
        Learning rate for un-matched / head parameters.
    scale_dict:
        Mapping of ``{prefix: scale_factor}``.  The first matching prefix
        wins (most-specific prefix should appear first if there is overlap).

    Returns
    -------
    List[dict]
        A list of param-group dicts ready to pass to any ``torch.optim``
        optimizer.

    Example
    -------
    >>> groups = layer_lr_groups(
    ...     model,
    ...     base_lr=1e-4,
    ...     scale_dict={"encoder": 0.01, "trunk": 0.1},
    ... )
    >>> opt = torch.optim.Adam(groups)
    """
    # Build: prefix -> list[param]
    bucket: Dict[str, List] = {p: [] for p in scale_dict}
    default: List = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for prefix in scale_dict:
            if _matches_any(name, [prefix]):
                bucket[prefix].append(param)
                matched = True
                break
        if not matched:
            default.append(param)

    groups: List[dict] = []
    for prefix, params in bucket.items():
        if params:
            groups.append({"params": params, "lr": base_lr * scale_dict[prefix]})
    if default:
        groups.append({"params": default, "lr": base_lr})
    return groups


def count_trainable(model: nn.Module) -> Dict[str, int]:
    """Count trainable, frozen, and total parameters.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    dict
        ``{"trainable": int, "frozen": int, "total": int}``

    Example
    -------
    >>> stats = count_trainable(model)
    >>> print(stats)
    {'trainable': 12500, 'frozen': 37500, 'total': 50000}
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
