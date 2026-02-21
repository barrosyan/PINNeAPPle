from __future__ import annotations
"""Utility functions for graph neural network operations."""
import torch


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    src: (B, E, F)
    index: (E,) destination indices [0..N-1]
    returns: (B, N, F) where output[:, index[e], :] += src[:, e, :]
    """
    B, E, F = src.shape
    out = torch.zeros((B, dim_size, F), device=src.device, dtype=src.dtype)
    out.index_add_(1, index, src)
    return out
