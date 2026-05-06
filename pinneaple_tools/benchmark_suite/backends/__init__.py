from __future__ import annotations

from .base import Backend
from .native_pinn import NativePINNBackend
from .pinneaple_models_backend import PinneapleModelsBackend
from .physicsnemo_sym import PhysicsNeMoSymBackend
from .deepxde_backend import DeepXDEBackend
from .jax_pinn import JAXPINNBackend

__all__ = [
    "Backend",
    "NativePINNBackend",
    "PinneapleModelsBackend",
    "PhysicsNeMoSymBackend",
    "DeepXDEBackend",
    "JAXPINNBackend",
]