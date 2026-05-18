"""Backend abstraction layer for PINNeAPPle.

Provides a simple global backend selector so PINN code can switch between
PyTorch (default) and JAX without modifying the calling code.
"""
from __future__ import annotations

from typing import Literal

BackendName = Literal["torch", "jax"]


class Backend:
    """Namespace for backend string constants."""
    TORCH: str = "torch"
    JAX: str = "jax"


_current_backend: str = Backend.TORCH


def set_backend(name: str) -> None:
    """Set the active computation backend.

    Parameters
    ----------
    name:
        Either ``"torch"`` or ``"jax"``.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend string.
    """
    global _current_backend
    if name not in (Backend.TORCH, Backend.JAX):
        raise ValueError(
            f"Unknown backend '{name}'. Choose 'torch' or 'jax'."
        )
    _current_backend = name


def get_backend() -> str:
    """Return the name of the currently active backend."""
    return _current_backend
