from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Union
import torch


@dataclass
class ChunkSpec:
    """
    Generic chunking specification for tensors.

    Backward-compatible fields:
      - mode: "none" | "time" | "spatial"  (treated as aliases for axis chunking)
      - chunk_size: chunk length along the chosen axis
      - dim: axis to chunk along (default 0)

    Generic extension:
      - dims: optional multiple axes to chunk along (cartesian tiling)
      - chunk_sizes: optional per-axis chunk sizes (matches dims length)

    Rules:
      - If mode == "none": yields the full tensor.
      - Else, if dims/chunk_sizes are provided: multi-axis chunking.
      - Else: single-axis chunking using dim/chunk_size.
    """
    mode: str = "none"
    chunk_size: int = 1024
    dim: int = 0

    # Optional generic multi-axis chunking (doesn't break old usage)
    dims: Sequence[int] = ()
    chunk_sizes: Sequence[int] = ()


def iter_chunks(x: torch.Tensor, spec: ChunkSpec) -> Iterator[torch.Tensor]:
    if spec.mode == "none":
        yield x
        return

    # -------- multi-axis (generic extension) --------
    if spec.dims and spec.chunk_sizes:
        dims = list(spec.dims)
        sizes = [int(s) for s in spec.chunk_sizes]
        if len(dims) != len(sizes):
            raise ValueError("ChunkSpec.dims and ChunkSpec.chunk_sizes must have the same length.")

        def rec(t: torch.Tensor, depth: int = 0) -> Iterator[torch.Tensor]:
            if depth == len(dims):
                yield t
                return

            d = dims[depth]
            n = t.shape[d]
            cs = sizes[depth]

            for i in range(0, n, cs):
                sl = [slice(None)] * t.ndim
                sl[d] = slice(i, min(i + cs, n))
                yield from rec(t[tuple(sl)], depth + 1)

        yield from rec(x)
        return

    # -------- single-axis (backward compatible) --------
    # mode "time"/"spatial" are treated as aliases; behavior is controlled by dim.
    d = int(spec.dim)
    n = x.shape[d]
    cs = int(spec.chunk_size)

    for i in range(0, n, cs):
        sl = [slice(None)] * x.ndim
        sl[d] = slice(i, min(i + cs, n))
        yield x[tuple(sl)]