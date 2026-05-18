from __future__ import annotations
"""Preset registry — problem specifications for common physics domains."""

from .registry import get_preset, list_presets, register_preset
from .terramechanics import TerramechanicsResiduals, bekker_wong_surrogate_2d

__all__ = [
    "get_preset",
    "list_presets",
    "register_preset",
    "TerramechanicsResiduals",
    "bekker_wong_surrogate_2d",
]
