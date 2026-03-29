"""Registry mapping problem preset names to factory functions.

Usage
-----
from pinneaple_environment.presets.registry import get_preset, list_presets

spec = get_preset("burgers_1d", nu=0.02)
spec = get_preset("ns_incompressible_2d", Re=200.0)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..spec import ProblemSpec

_REGISTRY: Dict[str, Callable[..., ProblemSpec]] = {}


def register_preset(name: str):
    """Decorator to register a preset factory function by name."""
    def deco(fn: Callable[..., ProblemSpec]) -> Callable[..., ProblemSpec]:
        key = str(name).lower().strip()
        _REGISTRY[key] = fn
        return fn
    return deco


def get_preset(name: str, **kwargs: Any) -> ProblemSpec:
    """Instantiate a problem preset by name.

    Parameters
    ----------
    name : preset identifier (case-insensitive)
    **kwargs : passed to the preset factory (e.g. nu=0.01, Re=200)

    Raises
    ------
    KeyError if the name is not registered.
    """
    key = str(name).lower().strip()
    if key not in _REGISTRY:
        _auto_register()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown problem preset '{name}'. Available: {list_presets()}"
        )
    return _REGISTRY[key](**kwargs)


def list_presets() -> List[str]:
    """Return sorted list of all registered preset names."""
    _auto_register()
    return sorted(_REGISTRY.keys())


def _auto_register() -> None:
    """Import preset modules so their decorators execute."""
    from . import academics as _ac  # noqa: F401
    from . import cfd as _cfd  # noqa: F401
    try:
        from . import industry as _ind  # noqa: F401
    except Exception:
        pass
    try:
        from . import structural as _struct  # noqa: F401
    except Exception:
        pass
    try:
        from . import engineering as _eng  # noqa: F401
    except Exception:
        pass
    try:
        from . import multidisciplinary as _multi  # noqa: F401
    except Exception:
        pass
    try:
        from . import solid_mechanics as _sm  # noqa: F401
    except Exception:
        pass
