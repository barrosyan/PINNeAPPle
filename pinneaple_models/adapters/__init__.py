from __future__ import annotations

"""Model adapters.

Adapters provide a uniform way to call models given a dict-batch, without
requiring every existing model implementation to be refactored at once.

The Arena can select an adapter automatically from a ModelSpec.
"""

from .base import ModelAdapter, select_adapter
from .pinn import PINNAdapter
from .operators import OperatorAdapter
from .gnn import GNNAdapter
from .ts import TimeSeriesAdapter
from .ae import AutoEncoderAdapter

__all__ = [
    "ModelAdapter",
    "select_adapter",
    "PINNAdapter",
    "OperatorAdapter",
    "GNNAdapter",
    "TimeSeriesAdapter",
    "AutoEncoderAdapter",
]
