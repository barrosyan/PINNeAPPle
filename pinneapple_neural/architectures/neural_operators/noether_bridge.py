"""Noether (Emmi AI) architecture bridge for PINNeAPPle.

Wraps the transformer-based neural surrogate architectures from
``emmiai-noether`` (UPT, AB-UPT, Transolver, Transformer, and their
aerodynamics variants) as ``NeuralOperatorBase`` subclasses so that they
are usable through PINNeAPPle's ``NeuralOperatorCatalog`` and can be
trained with the standard ``Trainer`` or the dedicated
``NoetherSurrogateTrainer``.

Install the optional dependency first:
    pip install emmiai-noether

Model overview
--------------
NoetherUPT          — Universal Physics Transformer (arXiv:2402.12365)
NoetherABUPT        — Anchored-Branched UPT (arXiv:2502.09692)
NoetherTransformer  — vanilla transformer backbone (Noether wrapper)
NoetherTransolver   — Physics-Attention Transolver (arXiv:2402.02366)
NoetherAeroUPT      — UPT with CFD surface/volume domain routing
NoetherAeroABUPT    — AB-UPT with CFD domain routing
NoetherAeroTransformer — Transformer for CFD aerodynamics
NoetherAeroTransolver  — Transolver for CFD aerodynamics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from .base import NeuralOperatorBase, OperatorOutput


# ---------------------------------------------------------------------------
# Shared config base
# ---------------------------------------------------------------------------

@dataclass
class NoetherModelConfig:
    """Minimal config wrapper forwarded verbatim to noether's own config classes.

    Parameters
    ----------
    model_name : one of the Noether model keys in ``_NOETHER_CLS`` below.
    config_kwargs : keyword arguments forwarded to the matching
        ``noether.core.schemas.models.<ModelName>Config``.
    device : torch device string; default "cuda" if available else "cpu".
    """
    model_name: str = "upt"
    config_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _import_noether():
    try:
        import noether  # noqa: F401
    except ImportError:
        raise ImportError(
            "emmiai-noether is required for Noether model wrappers.\n"
            "Install with:  pip install emmiai-noether\n"
            "or from source: https://github.com/Emmi-AI/noether"
        )


def _get_noether_cls(name: str):
    _import_noether()
    from noether.modeling.models import (
        UPT,
        AnchoredBranchedUPT,
        Transformer,
        Transolver,
        AeroUPT,
        AeroABUPT,
        AeroTransformer,
        AeroTransolver,
    )
    _MAP = {
        "upt": UPT,
        "abupt": AnchoredBranchedUPT,
        "anchored_branched_upt": AnchoredBranchedUPT,
        "transformer": Transformer,
        "transolver": Transolver,
        "aero_upt": AeroUPT,
        "aero_abupt": AeroABUPT,
        "aero_transformer": AeroTransformer,
        "aero_transolver": AeroTransolver,
    }
    key = name.lower().strip()
    if key not in _MAP:
        raise KeyError(f"Unknown Noether model '{name}'. Available: {sorted(_MAP)}")
    return _MAP[key]


def _get_noether_cfg_cls(name: str):
    """Return the corresponding Noether config schema class, if it exists."""
    _import_noether()
    try:
        from noether.core.schemas import models as _model_schemas
        cfg_name = {
            "upt": "UPTConfig",
            "abupt": "ABUPTConfig",
            "anchored_branched_upt": "ABUPTConfig",
            "transformer": "TransformerConfig",
            "transolver": "TransolverConfig",
            "aero_upt": "AeroUPTConfig",
            "aero_abupt": "AeroABUPTConfig",
            "aero_transformer": "AeroTransformConfig",
            "aero_transolver": "AeroTransolverConfig",
        }.get(name.lower().strip())
        if cfg_name and hasattr(_model_schemas, cfg_name):
            return getattr(_model_schemas, cfg_name)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class _NoetherWrapper(NeuralOperatorBase):
    """Internal wrapper that adapts any Noether model to NeuralOperatorBase.

    The inner model is instantiated lazily on first call to ``forward()`` so
    that ``import`` of this module does not fail when emmiai-noether is absent.

    Forward contract
    ----------------
    Accepts the same keyword arguments that the underlying Noether model's
    ``forward()`` accepts (pass them through **kwargs). Returns an
    ``OperatorOutput`` with:
      - y       : the raw tensor output from the Noether model
      - losses  : empty dict (losses are computed externally by NoetherSurrogateTrainer)
      - extras  : {"noether_raw": raw_output} so downstream code can inspect it
    """

    def __init__(self, noether_model_name: str, **config_kwargs) -> None:
        super().__init__()
        self._model_name = noether_model_name
        self._config_kwargs = config_kwargs
        self._inner: Optional[torch.nn.Module] = None

    def _build(self) -> None:
        if self._inner is not None:
            return
        model_cls = _get_noether_cls(self._model_name)
        cfg_cls = _get_noether_cfg_cls(self._model_name)
        if cfg_cls is not None and self._config_kwargs:
            cfg = cfg_cls(**self._config_kwargs)
            self._inner = model_cls(config=cfg)
        elif self._config_kwargs:
            self._inner = model_cls(**self._config_kwargs)
        else:
            self._inner = model_cls()
        # Move params to the same device as this module
        try:
            device = next(self.parameters()).device
            self._inner = self._inner.to(device)
        except StopIteration:
            pass

    def forward(self, *args, **kwargs) -> OperatorOutput:
        self._build()
        raw = self._inner(*args, **kwargs)  # type: ignore[operator]
        if isinstance(raw, torch.Tensor):
            y = raw
        elif hasattr(raw, "y"):
            y = raw.y
        elif isinstance(raw, dict):
            # Noether models sometimes return {field: tensor}
            y = torch.stack(list(raw.values()), dim=-1)
        else:
            y = torch.as_tensor(raw)
        return OperatorOutput(y=y, losses={}, extras={"noether_raw": raw})

    @property
    def inner_model(self) -> Optional[torch.nn.Module]:
        """Access the underlying Noether model (built lazily)."""
        self._build()
        return self._inner


# ---------------------------------------------------------------------------
# Named subclasses (for type-checking clarity and registry registration)
# ---------------------------------------------------------------------------

class NoetherUPT(_NoetherWrapper):
    """Universal Physics Transformer (Noether/Emmi AI).

    Reference: arXiv:2402.12365
    Suitable for: mesh-based CFD surrogates with unstructured geometry.

    Parameters forwarded to ``UPTConfig``:
        dim, num_heads, depth, supernode_k, perceiver_latents, ...
    """
    def __init__(self, **config_kwargs):
        super().__init__("upt", **config_kwargs)


class NoetherABUPT(_NoetherWrapper):
    """Anchored-Branched Universal Physics Transformer (Noether/Emmi AI).

    Reference: arXiv:2502.09692
    Suitable for: multi-domain CFD (surface + volume) with KV-caching.

    Parameters forwarded to ``ABUPTConfig``.
    """
    def __init__(self, **config_kwargs):
        super().__init__("abupt", **config_kwargs)


class NoetherTransformer(_NoetherWrapper):
    """Vanilla Transformer backbone from Noether.

    Parameters forwarded to ``TransformerConfig``.
    """
    def __init__(self, **config_kwargs):
        super().__init__("transformer", **config_kwargs)


class NoetherTransolver(_NoetherWrapper):
    """Transolver / Transolver++ Physics-Attention backbone (Noether/Emmi AI).

    Reference: arXiv:2402.02366 / arXiv:2502.02414
    Suitable for: operator learning with physics-guided slice attention.

    Parameters forwarded to ``TransolverConfig``.
    """
    def __init__(self, **config_kwargs):
        super().__init__("transolver", **config_kwargs)


class NoetherAeroUPT(_NoetherWrapper):
    """AeroUPT — UPT with surface/volume domain routing for aerodynamics."""
    def __init__(self, **config_kwargs):
        super().__init__("aero_upt", **config_kwargs)


class NoetherAeroABUPT(_NoetherWrapper):
    """AeroABUPT — AB-UPT with surface/volume domain routing."""
    def __init__(self, **config_kwargs):
        super().__init__("aero_abupt", **config_kwargs)


class NoetherAeroTransformer(_NoetherWrapper):
    """AeroTransformer — Transformer for CFD aerodynamics."""
    def __init__(self, **config_kwargs):
        super().__init__("aero_transformer", **config_kwargs)


class NoetherAeroTransolver(_NoetherWrapper):
    """AeroTransolver — Transolver for CFD aerodynamics."""
    def __init__(self, **config_kwargs):
        super().__init__("aero_transolver", **config_kwargs)


# ---------------------------------------------------------------------------
# Registry entries (consumed by neural_operators/registry.py)
# ---------------------------------------------------------------------------

NOETHER_REGISTRY: Dict[str, type] = {
    # UPT family
    "noether_upt": NoetherUPT,
    "upt": NoetherUPT,
    "universal_physics_transformer": NoetherUPT,
    # AB-UPT family
    "noether_abupt": NoetherABUPT,
    "abupt": NoetherABUPT,
    "anchored_branched_upt": NoetherABUPT,
    # Transformer / Transolver backbones
    "noether_transformer": NoetherTransformer,
    "noether_transolver": NoetherTransolver,
    "transolver": NoetherTransolver,
    # Aerodynamics wrappers
    "noether_aero_upt": NoetherAeroUPT,
    "aero_upt": NoetherAeroUPT,
    "noether_aero_abupt": NoetherAeroABUPT,
    "aero_abupt": NoetherAeroABUPT,
    "noether_aero_transformer": NoetherAeroTransformer,
    "aero_transformer": NoetherAeroTransformer,
    "noether_aero_transolver": NoetherAeroTransolver,
    "aero_transolver": NoetherAeroTransolver,
}
