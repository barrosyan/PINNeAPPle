"""Multi-backend support for PINNeAPPle: PyTorch (default) + JAX.

Provides a backend abstraction layer so PINN code can run on PyTorch or JAX
without modification.  JAX gives ``vmap``/``jit`` speedups on CPU/TPU; the
default PyTorch backend requires no extra dependencies.

Quick start::

    from pinneapple_tools.compute_backends import set_backend, get_backend, JAXBackend

    # Switch to JAX (requires: pip install jax jaxlib)
    set_backend("jax")
    print(get_backend())  # "jax"

    # JIT-compile a PINN. residual_fn receives (u_fn, x), where u_fn is
    # model_fn bound to params (x -> u(x)), so it can take derivatives via
    # jax.grad/jax.hessian -- not a precomputed value, which JAX cannot
    # differentiate.
    compiled = JAXBackend.jit_pinn(model_fn, residual_fn)

    # Vectorise a single-point residual over a batch of collocation points
    batched = JAXBackend.vmap_residual(single_pt_residual)

    # Convert between PyTorch and JAX
    jax_x = JAXBackend.torch_to_jax(torch_tensor)
    torch_x = JAXBackend.jax_to_torch(jax_array)
"""

from .backend import Backend, get_backend, set_backend
from .jax_backend import JAXBackend, jax_available, jax_pinn, jit_pinn, vmap_residual

__all__ = [
    # Backend registry
    "Backend",
    "get_backend",
    "set_backend",
    # JAX utilities
    "JAXBackend",
    "jax_available",
    "jax_pinn",
    "jit_pinn",
    "vmap_residual",
]
