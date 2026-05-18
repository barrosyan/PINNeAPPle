from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 10.0
    w_data: float = 1.0