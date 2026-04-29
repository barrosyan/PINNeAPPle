"""Multi-backend support for PINNeAPPle: PyTorch (default) + JAX.

Provides a backend abstraction layer so PINN code can run on PyTorch or JAX
without modification.  JAX gives ``vmap``/``jit`` speedups on CPU/TPU; the
default PyTorch backend requires no extra dependencies.

Quick start::

    from pinneaple_backend import set_backend, get_backend, JAXBackend

    # Switch to JAX (requires: pip install jax jaxlib)
    set_backend("jax")
    print(get_backend())  # "jax"

    # JIT-compile a PINN (model_fn + residual_fn must use JAX arrays)
    compiled = JAXBackend.jit_pinn(model_fn, residual_fn)

    # Vectorise a single-point residual over a batch
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
