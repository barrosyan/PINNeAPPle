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
    """Return ``True`` if JAX can be imported.

    As a side effect, enables JAX's 64-bit precision (``jax_enable_x64``).
    JAX defaults to float32 and silently downcasts float64 inputs, which
    would otherwise turn a PyTorch float64 tensor (the common choice for
    PINNs, since second-derivative PDE residuals are precision-sensitive)
    into a float32 JAX array with no warning.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
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
            Callable ``(params, x) -> u`` mapping parameters + a single
            collocation point to the (scalar) PDE solution there.
        residual_fn:
            Callable ``(u_fn, x) -> r`` computing the PDE residual, where
            ``u_fn`` is ``model_fn`` partially applied to ``params`` (i.e.
            ``x -> u(x)``) rather than an already-evaluated array. Unlike
            PyTorch, JAX has no autograd tape attached to a computed value,
            so derivatives must be taken of a *function*
            (``jax.grad``/``jax.jacfwd``/``jax.hessian``) -- passing a
            precomputed ``u`` would make every derivative term silently
            evaluate to zero. For example, the 1D Poisson residual
            ``u'' = -pi**2 sin(pi x)`` at a single collocation point::

                def residual_fn(u_fn, x):
                    u_x  = jax.grad(u_fn)(x)
                    u_xx = jax.grad(lambda x: jax.grad(u_fn)(x))(x)
                    return u_xx + jnp.pi ** 2 * jnp.sin(jnp.pi * x)

            and a 2D Laplacian ``u_xx + u_yy`` (e.g. for a biharmonic-type
            PDE) via ``jax.hessian``::

                def residual_fn(u_fn, x):
                    hess = jax.hessian(u_fn)(x)   # shape (2, 2)
                    return jnp.trace(hess)

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
            u_fn = lambda x: model_fn(params, x)
            return residual_fn(u_fn, x)

        return _compiled

    @staticmethod
    def vmap_residual(residual_fn: Callable, batch_size: int = 64) -> Callable:
        """Vectorise *residual_fn* over the leading batch dimension.

        Parameters
        ----------
        residual_fn:
            A function that operates on a **single** collocation point,
            ``(u_fn, x_i) -> r_i``, where ``u_fn`` is the (non-batched)
            model closure ``x -> u(x)`` -- see :meth:`jit_pinn`.
        batch_size:
            Unused; retained for API compatibility (JAX's ``vmap`` infers
            batch size automatically from input shapes).

        Returns
        -------
        Callable
            The vectorised ``(u_fn, x_batch) -> r_batch`` function. ``u_fn``
            is shared across the batch (not mapped); only ``x`` is.
        """
        if not jax_available():
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        import jax

        return jax.vmap(residual_fn, in_axes=(None, 0))

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

        This is a non-differentiable boundary: the tensor is detached
        before conversion, so any PyTorch autograd graph attached to it is
        severed and gradients will not flow back through a
        torch -> jax -> torch round trip. Use it for array *data* (e.g.
        collocation points, evaluated results), not as a step inside a
        pipeline that needs end-to-end differentiability.

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

        Like :meth:`torch_to_jax`, this is a non-differentiable boundary:
        the resulting tensor has no grad history, even if *jax_array* was
        produced by a JAX computation that is itself differentiable (use
        ``jax.grad``/``jax.jacfwd`` on the JAX side before converting).

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
        ``(params, x) -> u`` mapping parameters + a single collocation
        point to the (scalar) PDE solution there.
    residual_fn:
        ``(u_fn, x) -> r`` computing the PDE residual at a single point via
        derivatives of ``u_fn`` (see :meth:`JAXBackend.jit_pinn`).
    vectorise:
        Whether to wrap *residual_fn* with ``vmap`` before JIT, batching
        over the leading dimension of ``x``.

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
