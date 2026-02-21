"""Base SolverOutput and SolverBase for numerical solvers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SolverOutput:
    """
    Standard output for solvers / numerical methods.
    result:
      - main output tensor (spectra, modes, field solution, etc.)
    extras:
      - metadata (freqs, imfs, diagnostics, residuals)
    """
    result: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class SolverBase(nn.Module):
    """Base class for numerical solvers."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mean((a - b) ** 2)
