"""Thread-safe LRU cache for in-memory data with configurable capacity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional
from collections import OrderedDict
import threading


@dataclass
class CacheStats:
    """
    Statistics container for the LRUCache.

    Attributes
    ----------
    hits : int
        Number of successful cache retrievals.
    misses : int
        Number of failed cache retrievals.
    evictions : int
        Number of items removed due to capacity limits.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class LRUCache:
    """
    Thread-safe LRU cache.

    Notes:
      - max_items is the primary control
      - you can add your own "weigher" if you want max_bytes later
    """

    def __init__(self, max_items: int = 256):
        """
        Initialize the LRUCache.

        Parameters
        ----------
        max_items : int
            Maximum number of items allowed in the cache.
        """
        self.max_items = int(max_items)
        self._od: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        If the key exists, it is marked as recently used.

        Parameters
        ----------
        key : Hashable
            The key associated with the cached value.

        Returns
        -------
        Optional[Any]
            The cached value if present, otherwise None.
        """
        with self._lock:
            if key in self._od:
                self._od.move_to_end(key)
                self.stats.hits += 1
                return self._od[key]
            self.stats.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        """
        Insert or update a value in the cache.

        If inserting a new item causes the cache to exceed its
        capacity, the least recently used item is evicted.

        Parameters
        ----------
        key : Hashable
            The key to store.
        value : Any
            The value to associate with the key.
        """
        with self._lock:
            if key in self._od:
                self._od[key] = value
                self._od.move_to_end(key)
                return
            self._od[key] = value
            self._od.move_to_end(key)

            while len(self._od) > self.max_items:
                self._od.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        """
        Remove all items from the cache and reset statistics.
        """
        with self._lock:
            self._od.clear()
            self.stats = CacheStats()

    def __len__(self) -> int:
        """
        Return the number of items currently stored in the cache.

        Returns
        -------
        int
            The number of cached entries.
        """
        with self._lock:
            return len(self._od)