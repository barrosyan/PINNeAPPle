"""Neighbor search utilities (grid hashing) for particle methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class NeighborList:
    """Compressed neighbor list representation."""

    # neighbor indices concatenated
    j: torch.Tensor  # (E,)
    # pointer offsets per particle (CSR)
    indptr: torch.Tensor  # (N+1,)


def build_neighbor_list(
    pos: torch.Tensor,
    *,
    h: float,
    max_neighbors: int = 64,
) -> NeighborList:
    """Build neighbor list for 2D/3D particles using uniform grid.

    Parameters
    ----------
    pos: (N,D)
    h: smoothing length (cell size ~ h)
    max_neighbors: cap per particle (MVP safety)
    """
    if pos.ndim != 2:
        raise ValueError("pos must be (N,D)")
    N, D = pos.shape
    device = pos.device
    h = float(h)
    if h <= 0:
        raise ValueError("h must be > 0")

    # grid coords
    g = torch.floor(pos / h).to(torch.int64)  # (N,D)
    # hash
    if D == 2:
        key = g[:, 0] * 73856093 ^ g[:, 1] * 19349663
    elif D == 3:
        key = g[:, 0] * 73856093 ^ g[:, 1] * 19349663 ^ g[:, 2] * 83492791
    else:
        raise ValueError("Only D=2 or D=3 supported")

    order = torch.argsort(key)
    key_sorted = key[order]
    g_sorted = g[order]

    # find cell boundaries
    unique_keys, counts = torch.unique_consecutive(key_sorted, return_counts=True)
    cell_starts = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=torch.int64), counts[:-1]]), dim=0)
    cell_ends = cell_starts + counts

    # map key->(start,end) using dict on CPU for MVP
    # (for large N, you'd do this fully on GPU; MVP is fine here)
    mapping = {int(k.item()): (int(s.item()), int(e.item())) for k, s, e in zip(unique_keys.cpu(), cell_starts.cpu(), cell_ends.cpu())}

    # neighbor offsets (3^D cells)
    if D == 2:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    else:
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    # build CSR
    indptr = torch.zeros((N + 1,), device=device, dtype=torch.int64)
    neighbors = []

    pos_cpu = pos.detach().cpu()
    g_cpu = g.detach().cpu()
    order_cpu = order.detach().cpu()
    g_sorted_cpu = g_sorted.detach().cpu()

    for i in range(N):
        gi = g_cpu[i]
        cand = []
        for off in offsets:
            gj = gi + torch.tensor(off, dtype=torch.int64)
            if D == 2:
                k = int((gj[0] * 73856093 ^ gj[1] * 19349663).item())
            else:
                k = int((gj[0] * 73856093 ^ gj[1] * 19349663 ^ gj[2] * 83492791).item())
            if k in mapping:
                s, e = mapping[k]
                cand.extend(order_cpu[s:e].tolist())
        if not cand:
            indptr[i + 1] = indptr[i]
            continue
        cand = list(dict.fromkeys(cand))  # unique preserve order
        # distance filter
        pi = pos_cpu[i]
        js = []
        for j in cand:
            if j == i:
                continue
            pj = pos_cpu[j]
            if torch.norm(pi - pj).item() <= 2.0 * h:
                js.append(int(j))
            if len(js) >= max_neighbors:
                break
        neighbors.extend(js)
        indptr[i + 1] = indptr[i] + len(js)

    j = torch.tensor(neighbors, device=device, dtype=torch.int64)
    return NeighborList(j=j, indptr=indptr)
