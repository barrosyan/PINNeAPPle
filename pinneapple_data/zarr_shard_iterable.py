"""Shard-aware IterableDataset for balanced iteration over multi-shard Zarr UPD stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import os
import json

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .device import pin_sample, to_device_sample
from .zarr_cached_store_bytes import CachedUPDZarrStoreBytes, ZarrByteCacheConfig


@dataclass
class ShardAwareConfig:
    """
    Configuration for shard-aware iteration behavior.

    Attributes
    ----------
    pin_memory : bool
        Whether to pin tensors in CPU pinned memory. Default is True.
    target_device : str
        Target device ("cpu" or "cuda"). Default is "cpu".
    transfer_non_blocking : bool
        Whether to use non-blocking GPU transfers. Default is True.
    use_sample_cache : bool
        Whether to use per-sample caching. Default is True.
    """
    pin_memory: bool = True
    target_device: str = "cpu"  # "cpu" or "cuda"
    transfer_non_blocking: bool = True
    use_sample_cache: bool = True


def _load_index(root: str) -> List[Tuple[str, int]]:
    """
    Returns a flat list of (zarr_path, count) from root/index.json
    """
    index_path = os.path.join(root, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    parts: List[Tuple[str, int]] = []
    shards = idx.get("shards", {})
    for _k, arr in shards.items():
        for item in arr:
            parts.append((os.path.join(root, item["path"]), int(item.get("count", 0))))
    return parts


def _greedy_balance(parts: List[Tuple[str, int]], num_bins: int) -> List[List[Tuple[str, int]]]:
    """
    Greedy bin packing by count (approx load).
    """
    bins: List[List[Tuple[str, int]]] = [[] for _ in range(num_bins)]
    loads = [0 for _ in range(num_bins)]

    parts_sorted = sorted(parts, key=lambda x: x[1], reverse=True)
    for p in parts_sorted:
        j = min(range(num_bins), key=lambda i: loads[i])
        bins[j].append(p)
        loads[j] += p[1]
    return bins


class ShardAwareZarrUPDIterable(IterableDataset):
    """
    Shard-aware iterator:
      - reads root/index.json
      - balances shard parts across DataLoader workers
      - iterates each assigned zarr store sequentially

    This scales well in industry:
      - natural partitioning
      - easy parallel reads
      - balanced work
    """
    def __init__(
        self,
        root: str,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        sample_ctor: Any = None,
        cache: Optional[ZarrByteCacheConfig] = None,
        cfg: Optional[ShardAwareConfig] = None,
    ):
        """
        Initialize the shard-aware iterable dataset.

        Parameters
        ----------
        root : str
            Root path containing index.json and shard directories.
        fields : Optional[Sequence[str]], optional
            Field names to load.
        coords : Optional[Sequence[str]], optional
            Coordinate names to load.
        dtype : Optional[torch.dtype], optional
            Target dtype for tensors.
        sample_ctor : Any, optional
            Optional sample constructor.
        cache : Optional[ZarrByteCacheConfig], optional
            Cache configuration.
        cfg : Optional[ShardAwareConfig], optional
            Iteration configuration.
        """
        super().__init__()
        self.root = root
        self.fields = fields
        self.coords = coords
        self.dtype = dtype
        self.sample_ctor = sample_ctor
        self.cache = cache or ZarrByteCacheConfig()
        self.cfg = cfg or ShardAwareConfig()

        self._parts = _load_index(root)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over samples from assigned shards with optional pinning and GPU transfer.

        Yields
        ------
        Any
            Sample from the store, optionally pinned and/or moved to target device.
        """
        wi = get_worker_info()
        if wi is None:
            assigned = self._parts
        else:
            bins = _greedy_balance(self._parts, wi.num_workers)
            assigned = bins[wi.id]

        do_cuda = self.cfg.target_device.lower().startswith("cuda")

        # Iterate all assigned shards
        for zarr_path, count in assigned:
            store = CachedUPDZarrStoreBytes(zarr_path, cache=self.cache, mode="r")
            n = store.count()

            # Prefer store.count() as source of truth
            for i in range(n):
                s = store.read_sample(
                    i,
                    fields=self.fields,
                    coords=self.coords,
                    device="cpu",
                    dtype=self.dtype,
                    sample_ctor=self.sample_ctor,
                    use_sample_cache=self.cfg.use_sample_cache,
                )
                if self.cfg.pin_memory:
                    s = pin_sample(s)
                if do_cuda:
                    s = to_device_sample(s, device="cuda", dtype=self.dtype, non_blocking=self.cfg.transfer_non_blocking)
                yield s
