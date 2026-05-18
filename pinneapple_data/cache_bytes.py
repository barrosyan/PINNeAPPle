"""Byte-constrained LRU cache for memory-bounded caching of tensors and samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Tuple
from collections import OrderedDict
import threading

import torch


@dataclass
class ByteCacheStats:
    """
    Statistics container for the ByteLRUCache.

    Attributes
    ----------
    hits : int
        Number of successful cache lookups.
    misses : int
        Number of failed cache lookups.
    evictions : int
        Number of items evicted due to memory constraints.
    bytes_in_use : int
        Current total memory usage (in bytes) of cached values.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_in_use: int = 0


def default_weigher(value: Any) -> int:
    """
    Estimate memory usage in bytes.

    - torch.Tensor: nbytes
    - dict with tensors: sum
    - PhysicalSample-like: sum(fields)+sum(coords)
    - fallback: 1 KB

    Parameters
    ----------
    value : Any
        The object whose memory footprint should be estimated.

    Returns
    -------
    int
        Estimated size in bytes.
    """
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())

    # PhysicalSample-like
    if hasattr(value, "fields") and hasattr(value, "coords"):
        total = 0
        for v in value.fields.values():
            total += default_weigher(v)
        for v in value.coords.values():
            total += default_weigher(v)
        return total

    if isinstance(value, dict):
        total = 0
        for v in value.values():
            total += default_weigher(v)
        return total

    return 1024


class ByteLRUCache:
    """
    Thread-safe LRU cache constrained by max_bytes (not max items).

    This cache limits total memory consumption instead of the number of
    stored items. When inserting new values causes the memory budget
    to be exceeded, the least recently used items are evicted until
    the cache fits within the configured byte limit.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024, weigher: Callable[[Any], int] = default_weigher):
        """
        Initialize the ByteLRUCache.

        Parameters
        ----------
        max_bytes : int
            Maximum total memory (in bytes) allowed for cached values.
        weigher : Callable[[Any], int]
            Function used to estimate the memory footprint of stored values.
        """
        self.max_bytes = int(max_bytes)
        self.weigher = weigher
        self._od: "OrderedDict[Hashable, Tuple[Any, int]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = ByteCacheStats()

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        If the key exists, the item is marked as recently used.

        Parameters
        ----------
        key : Hashable
            Cache key.

        Returns
        -------
        Optional[Any]
            The cached value if present, otherwise None.
        """
        with self._lock:
            if key in self._od:
                self._od.move_to_end(key)
                self.stats.hits += 1
                return self._od[key][0]
            self.stats.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        """
        Insert or update a value in the cache.

        If the key already exists, its value is replaced and memory
        accounting is updated. If the insertion causes the total
        memory usage to exceed `max_bytes`, least recently used
        items are evicted until the constraint is satisfied.

        Parameters
        ----------
        key : Hashable
            Cache key.
        value : Any
            Value to store.
        """
        w = int(self.weigher(value))

        with self._lock:
            if key in self._od:
                _, old_w = self._od[key]
                self.stats.bytes_in_use -= old_w
                self._od[key] = (value, w)
                self._od.move_to_end(key)
                self.stats.bytes_in_use += w
            else:
                self._od[key] = (value, w)
                self._od.move_to_end(key)
                self.stats.bytes_in_use += w

            # evict until within budget
            while self.stats.bytes_in_use > self.max_bytes and len(self._od) > 0:
                _, (_, ev_w) = self._od.popitem(last=False)
                self.stats.bytes_in_use -= ev_w
                self.stats.evictions += 1

    def clear(self) -> None:
        """
        Remove all items from the cache and reset statistics.
        """
        with self._lock:
            self._od.clear()
            self.stats = ByteCacheStats()

    def __len__(self) -> int:
        """
        Return the number of items currently stored in the cache.

        Returns
        -------
        int
            Number of cached entries.
        """
        with self._lock:
            return len(self._od)