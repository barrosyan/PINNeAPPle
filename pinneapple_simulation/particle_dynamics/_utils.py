"""Private utilities shared across pinneapple_dynamics backends."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _gravity_tensor(
    dim: int,
    gravity: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """Build a gravity vector tensor of length *dim*.

    Defaults to standard Earth gravity (9.81 m/s² downward along the y-axis
    for 2-D/3-D, i.e. index 1 is -9.81).
    """
    if gravity is None:
        g = [0.0] * dim
        if dim >= 2:
            g[1] = -9.81
    else:
        g = list(gravity)[:dim]
    return torch.tensor(g, dtype=torch.float32)
