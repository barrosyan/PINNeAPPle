from __future__ import annotations

from .base import Backend
from .native_pinn import NativePINNBackend
from .pinneapple_models_backend import PinneappleModelsBackend
from .physicsnemo_sym import PhysicsNeMoSymBackend
from .deepxde_backend import DeepXDEBackend
from .jax_pinn import JAXPINNBackend

__all__ = [
    "Backend",
    "NativePINNBackend",
    "PinneappleModelsBackend",
    "PhysicsNeMoSymBackend",
    "DeepXDEBackend",
    "JAXPINNBackend",
]