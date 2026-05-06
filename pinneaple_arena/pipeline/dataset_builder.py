"""Arena dataset builder — thin re-export from pinneaple_data.adapters.pinn_batch_builders.

Backwards-compatible alias: BundleDataLike == PINNBatch.
"""
from pinneaple_data.adapters.pinn_batch_builders import (  # noqa: F401
    PINNBatch,
    build_from_bundle,
    build_from_solver,
    build_from_real_data,
)

# Legacy alias kept for existing callers
BundleDataLike = PINNBatch

__all__ = [
    "BundleDataLike",
    "PINNBatch",
    "build_from_bundle",
    "build_from_solver",
    "build_from_real_data",
]
