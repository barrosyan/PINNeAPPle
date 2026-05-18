"""JAX backend utilities for PINNeAPPle PINNs.

Provides JIT compilation, vmap vectorisation, and array conversion helpers.
Falls back gracefully when JAX is not installed.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# JAX availability check
# ---------------------------------------------------------------------------


def jax_available() -> bool:
    """Return ``True`` if JAX can be imported."""
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# JAXBackend class
# ---------------------------------------------------------------------------


class JAXBackend:
    """JAX-based PINN utilities wrapping Equinox or Flax models.

    Provides:
    - :meth:`jit_pinn`        – JIT-compile a PINN forward + residual.
    - :meth:`vmap_residual`   – Vectorise residual over a batch dimension.
    - :meth:`grad_fn`         – Compute derivatives via ``jax.grad``.
    - :meth:`torch_to_jax`    – Convert a PyTorch tensor to a JAX array.
    - :meth:`jax_to_torch`    – Convert a JAX array to a PyTorch tensor.

    All methods raise :class:`ImportError` with a helpful message when JAX is
    not installed instead of crashing with an obscure ``ModuleNotFoundError``.
    """

    # ------------------------------------------------------------------
    # Transformation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def jit_pinn(model_fn: Callable, residual_fn: Callable) -> Callable:
        """JIT-compile ``model_fn`` and ``residual_fn`` into a single call.

        The compiled function signature is ``(params, x) -> residual``.

        Parameters
        ----------
        model_fn:
            Callable ``(params, x) -> u`` mapping parameters + collocation
            points to the PDE solution.
        residual_fn:
            Callable ``(u, x) -> r`` computing the PDE residual.

        Returns
        -------
        Callable
            A JIT-compiled ``(params, x) -> r`` function.
        """
        if not jax_available():
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        import jax

        @jax.jit
        def _compiled(params: Any, x: Any) -> Any:
            u = model_fn(params, x)
            return residual_fn(u, x)

        return _compiled

    @staticmethod
    def vmap_residual(residual_fn: Callable, batch_size: int = 64) -> Callable:
        """Vectorise *residual_fn* over the leading batch dimension.

        Parameters
        ----------
        residual_fn:
            A function that operates on a **single** collocation point
            ``(u_i, x_i) -> r_i``.
        batch_size:
            Unused; retained for API compatibility (JAX's ``vmap`` infers
            batch size automatically from input shapes).

        Returns
        -------
        Callable
            The vectorised ``(u_batch, x_batch) -> r_batch`` function.
        """
        if not jax_available():
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        import jax

        return jax.vmap(residual_fn)

    @staticmethod
    def grad_fn(fn: Callable, argnums: int = 0) -> Callable:
        """Return the gradient of *fn* with respect to argument *argnums*.

        Parameters
        ----------
        fn:
            Scalar-valued function.
        argnums:
            Which positional argument to differentiate (default: 0).

        Returns
        -------
        Callable
            The gradient function.
        """
        if not jax_available():
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        import jax

        return jax.grad(fn, argnums=argnums)

    # ------------------------------------------------------------------
    # Array conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def torch_to_jax(tensor: Any) -> Any:
        """Convert a PyTorch tensor to a JAX array.

        Parameters
        ----------
        tensor:
            A ``torch.Tensor`` (on any device; moved to CPU first).

        Returns
        -------
        jax.numpy.ndarray
        """
        if not jax_available():
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        import jax.numpy as jnp

        return jnp.array(tensor.detach().cpu().numpy())

    @staticmethod
    def jax_to_torch(jax_array: Any, device: str = "cpu") -> Any:
        """Convert a JAX array to a PyTorch tensor.

        Parameters
        ----------
        jax_array:
            Any JAX array.
        device:
            Target PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).

        Returns
        -------
        torch.Tensor
        """
        import numpy as np
        import torch

        return torch.from_numpy(np.array(jax_array)).to(device)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def jit_pinn(model_fn: Callable, residual_fn: Callable) -> Callable:
    """Module-level alias for :meth:`JAXBackend.jit_pinn`."""
    return JAXBackend.jit_pinn(model_fn, residual_fn)


def vmap_residual(residual_fn: Callable, batch_size: int = 64) -> Callable:
    """Module-level alias for :meth:`JAXBackend.vmap_residual`."""
    return JAXBackend.vmap_residual(residual_fn, batch_size=batch_size)


def jax_pinn(model_fn: Callable, residual_fn: Callable,
             vectorise: bool = True) -> Callable:
    """Convenience: JIT-compile and optionally vmap a PINN residual.

    Equivalent to::

        jit_pinn(model_fn, vmap_residual(residual_fn))

    when *vectorise* is ``True``.

    Parameters
    ----------
    model_fn:
        ``(params, x) -> u``
    residual_fn:
        ``(u, x) -> r``  (single-point or batched depending on *vectorise*)
    vectorise:
        Whether to wrap *residual_fn* with ``vmap`` before JIT.

    Returns
    -------
    Callable
        Compiled ``(params, x) -> r`` function.
    """
    if not jax_available():
        raise ImportError(
            "JAX is not installed. Install with: pip install jax jaxlib"
        )
    if vectorise:
        vmapped = JAXBackend.vmap_residual(residual_fn)
        return JAXBackend.jit_pinn(model_fn, vmapped)
    return JAXBackend.jit_pinn(model_fn, residual_fn)
