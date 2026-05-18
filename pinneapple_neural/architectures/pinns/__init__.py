from __future__ import annotations
"""PINN model family — all variants, base classes, and registry."""

from .base import PINNBase, PINNOutput
from .vanilla import VanillaPINN
from .inverse import InversePINN
from .pielm import PIELM
from .pinn_lstm import PINNLSTM
from .pinnsformer import PINNsFormer
from .vpinn import VPINN
from .xpinn import XPINN, SubnetWrapper
from .xtfc import XTFC, XTFCConfig, build_xtfc, tfc_available
from .registry import PINNCatalog

__all__ = [
    # base
    "PINNBase",
    "PINNOutput",
    # model variants
    "VanillaPINN",
    "InversePINN",
    "PIELM",
    "PINNLSTM",
    "PINNsFormer",
    "VPINN",
    "XPINN",
    "SubnetWrapper",
    "XTFC",
    "XTFCConfig",
    "build_xtfc",
    "tfc_available",
    # registry
    "PINNCatalog",
]
