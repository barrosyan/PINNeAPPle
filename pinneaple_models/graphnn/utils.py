from __future__ import annotations
"""Utility functions for graph neural network operations."""
import torch
from typing import List, Optional, Tuple

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

def scatter_add_batched(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Fixed/shared-graph batching.
    src: (B, E, F)
    index: (E,) destination indices [0..N-1]
    returns: (B, N, F) where out[:, index[e], :] += src[:, e, :]
    """
    B, E, F = src.shape
    out = torch.zeros((B, dim_size, F), device=src.device, dtype=src.dtype)
    out.index_add_(1, index, src)
    return out


def scatter_add_flat(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Flattened variable-graph batching.
    src: (E_total, F)
    index: (E_total,) destination indices [0..N_total-1]
    returns: (N_total, F) where out[index[e], :] += src[e, :]
    """
    F = src.shape[-1]
    out = torch.zeros((dim_size, F), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def flatten_graph_batch(
    x_list: List[torch.Tensor],
    edge_index_list: List[torch.Tensor],
    pos_list: Optional[List[torch.Tensor]] = None,
    edge_attr_list: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Turns per-sample graphs into a single flattened graph with node index offsets.

    x_list:        list of (N_b, Fin)
    edge_index_list: list of (2, E_b) with indices in [0..N_b-1]
    pos_list:      list of (N_b, P) or None
    edge_attr_list:list of (E_b, Fe) or None

    Returns:
      x_flat: (N_total, Fin)
      edge_index_flat: (2, E_total) with offsets applied
      pos_flat: (N_total, P) or None
      edge_attr_flat: (E_total, Fe) or None
      ptr: (B+1,) prefix sums over nodes (ptr[0]=0, ptr[b+1]=sum_{i<=b} N_i)
    """
    device = x_list[0].device
    dtype = x_list[0].dtype

    n_per = torch.tensor([x.shape[0] for x in x_list], device=device, dtype=torch.long)
    ptr = torch.zeros((len(x_list) + 1,), device=device, dtype=torch.long)
    ptr[1:] = torch.cumsum(n_per, dim=0)

    x_flat = torch.cat(x_list, dim=0)

    if pos_list is not None:
        pos_flat = torch.cat(pos_list, dim=0)
    else:
        pos_flat = None

    # Apply node offsets to each edge_index
    edge_chunks = []
    for b, ei in enumerate(edge_index_list):
        if ei.numel() == 0:
            continue
        off = ptr[b].item()
        edge_chunks.append(ei + off)
    edge_index_flat = torch.cat(edge_chunks, dim=1) if len(edge_chunks) > 0 else torch.zeros((2, 0), device=device, dtype=torch.long)

    if edge_attr_list is not None:
        edge_attr_flat = torch.cat(edge_attr_list, dim=0) if len(edge_attr_list) > 0 else torch.zeros((0, edge_attr_list[0].shape[-1]), device=device, dtype=dtype)
    else:
        edge_attr_flat = None

    return x_flat, edge_index_flat, pos_flat, edge_attr_flat, ptr