"""Model instantiation helpers.

Goal
----
Different model families historically used different constructor parameter
names (e.g. ``in_dim`` vs ``input_dim`` vs ``in_channels``).

The Arena needs to be able to instantiate models programmatically and compare
them consistently. This module normalizes kwargs and filters unsupported args
based on the model class signature.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from .base import BaseModel


@dataclass(frozen=True)
class InstantiateReport:
    """Diagnostics about how kwargs were normalized."""

    original: Dict[str, Any]
    normalized: Dict[str, Any]
    dropped: Dict[str, Any]
    used_aliases: Dict[str, str]


_ALIASES: Tuple[Tuple[str, str], ...] = (
    # canonical <- alias
    ("in_dim", "input_dim"),
    ("in_dim", "in_channels"),
    ("in_dim", "dim_in"),
    ("out_dim", "output_dim"),
    ("out_dim", "out_channels"),
    ("out_dim", "dim_out"),
    ("d_model", "width"),
)


def normalize_kwargs(kwargs: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Normalize user kwargs to a canonical set.

    Returns
    -------
    normalized : Dict[str, Any]
        Normalized kwargs (alias keys merged into canonical keys when missing).
    used_aliases : Dict[str, str]
        Map canonical_key -> alias_key that was used.
    """
    out = dict(kwargs)
    used: Dict[str, str] = {}

    for canonical, alias in _ALIASES:
        if canonical in out:
            continue
        if alias in out:
            out[canonical] = out.pop(alias)
            used[canonical] = alias

    return out, used


def _adapt_canonical_to_signature(model_cls: Type[Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Reverse of normalize_kwargs: some model classes (e.g. FourierNeuralOperator,
    whose __init__ requires in_channels/out_channels) only accept an ALIAS name,
    never the canonical one — normalize_kwargs alone renames alias->canonical, so
    for these classes the canonical key is simply absent from the signature and
    filter_supported_kwargs would silently drop it, leaving a required
    constructor arg unfilled (TypeError: missing N required positional arguments).
    If the canonical key isn't in the signature but one of its known aliases is,
    rename back to that alias before filtering."""
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    supported = {k for k in params if k != "self"}
    out = dict(kwargs)
    for canonical, alias in _ALIASES:
        if canonical in out and canonical not in supported and alias in supported and alias not in out:
            out[alias] = out.pop(canonical)
    return out


def filter_supported_kwargs(model_cls: Type[Any], kwargs: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep only kwargs accepted by ``model_cls.__init__``.

    If the constructor accepts ``**kwargs``, nothing is dropped.
    """
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters

    # accepts **kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs), {}

    supported = {k for k, p in params.items() if k not in ("self",)}
    kept: Dict[str, Any] = {}
    dropped: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in supported:
            kept[k] = v
        else:
            dropped[k] = v
    return kept, dropped


def instantiate(
    model_cls: Type[BaseModel],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
) -> Tuple[BaseModel, InstantiateReport]:
    """Instantiate a model class with normalized kwargs.

    Parameters
    ----------
    model_cls:
        Model class.
    kwargs:
        Keyword args. Aliases are normalized.
    strict:
        If True, raise if any kwargs are dropped.

    Returns
    -------
    model, report
    """
    kwargs = dict(kwargs or {})
    normalized, used_aliases = normalize_kwargs(kwargs)
    normalized = _adapt_canonical_to_signature(model_cls, normalized)
    kept, dropped = filter_supported_kwargs(model_cls, normalized)
    if strict and dropped:
        raise TypeError(
            f"Unsupported kwargs for {model_cls.__name__}: {sorted(dropped.keys())}. "
            f"Supported kwargs: {sorted(kept.keys())}."
        )
    model = model_cls(**kept)
    return model, InstantiateReport(original=dict(kwargs), normalized=dict(normalized), dropped=dropped, used_aliases=used_aliases)
