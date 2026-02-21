"""Byte-cached UPDZarrStore wrapper with configurable sample and field cache budgets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Hashable

import torch

from .zarr_store import UPDZarrStore
from .cache_bytes import ByteLRUCache


def _norm_keys(keys: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    Normalize a sequence of keys into a sorted tuple for cache key hashing.

    Parameters
    ----------
    keys : Optional[Sequence[str]]
        Key names or None.

    Returns
    -------
    Tuple[str, ...]
        Sorted tuple of string keys, or empty tuple if keys is None.
    """
    if keys is None:
        return tuple()
    return tuple(sorted([str(k) for k in keys]))


@dataclass
class ZarrByteCacheConfig:
    """
    Configuration for byte-bounded Zarr cache limits.

    Attributes
    ----------
    max_sample_bytes : int
        Maximum bytes for full-sample cache. Default is 512 MiB.
    max_field_bytes : int
        Maximum bytes for field-level cache. Default is 512 MiB.
    enable_field_cache : bool
        Whether to enable per-field caching. Default is True.
    """
    max_sample_bytes: int = 512 * 1024 * 1024   # full-sample cache budget
    max_field_bytes: int = 512 * 1024 * 1024    # field-level cache budget
    enable_field_cache: bool = True


class CachedUPDZarrStoreBytes:
    """
    Cached wrapper around UPDZarrStore using ByteLRUCache.
    """
    def __init__(self, root: str, *, cache: Optional[ZarrByteCacheConfig] = None, mode: str = "r"):
        """
        Initialize the byte-cached Zarr store.

        Parameters
        ----------
        root : str
            Root path of the underlying Zarr store.
        cache : Optional[ZarrByteCacheConfig]
            Cache configuration. Defaults to ZarrByteCacheConfig().
        mode : str, optional
            Access mode for the store. Default is "r".
        """
        self.store = UPDZarrStore(root, mode=mode)
        self.cache_cfg = cache or ZarrByteCacheConfig()

        self.sample_cache = ByteLRUCache(max_bytes=self.cache_cfg.max_sample_bytes)
        self.field_cache = ByteLRUCache(max_bytes=self.cache_cfg.max_field_bytes)

    def count(self) -> int:
        """Return the number of samples in the store."""
        return self.store.count()

    def manifest(self) -> Dict[str, Any]:
        """Return the store manifest dictionary."""
        return self.store.manifest()

    def _sample_key(
        self,
        i: int,
        fields: Optional[Sequence[str]],
        coords: Optional[Sequence[str]],
        device: str | torch.device,
        dtype: Optional[torch.dtype],
    ) -> Hashable:
        dev = str(device)
        dt = str(dtype) if dtype is not None else ""
        return ("sample", int(i), _norm_keys(fields), _norm_keys(coords), dev, dt)

    def _field_key(
        self,
        i: int,
        kind: str,
        name: str,
        device: str | torch.device,
        dtype: Optional[torch.dtype],
    ) -> Hashable:
        dev = str(device)
        dt = str(dtype) if dtype is not None else ""
        return (kind, int(i), str(name), dev, dt)

    def read_sample(
        self,
        i: int,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        device: str | torch.device = "cpu",
        dtype: Optional[torch.dtype] = None,
        sample_ctor=None,
        use_sample_cache: bool = True,
    ):
        """
        Read a sample by index with optional field-level byte caching.

        Parameters
        ----------
        i : int
            Sample index.
        fields : Optional[Sequence[str]], optional
            Field names to load. Default is all.
        coords : Optional[Sequence[str]], optional
            Coordinate names to load.
        device : str or torch.device, optional
            Target device. Default is "cpu".
        dtype : Optional[torch.dtype], optional
            Target dtype.
        sample_ctor : callable, optional
            Constructor for building sample from (fields, coords, meta).
        use_sample_cache : bool, optional
            Whether to use full-sample cache. Default is True.

        Returns
        -------
        Any
            PhysicalSample-like object or sample_ctor output.
        """
        sk = self._sample_key(i, fields, coords, device, dtype)
        if use_sample_cache:
            cached = self.sample_cache.get(sk)
            if cached is not None:
                return cached

        if self.cache_cfg.enable_field_cache:
            fk = self.store.field_keys()
            ck = self.store.coord_keys()
            req_fields = list(fields) if fields is not None else fk
            req_coords = list(coords) if coords is not None else ck

            out_fields: Dict[str, torch.Tensor] = {}
            out_coords: Dict[str, torch.Tensor] = {}

            for name in req_fields:
                k = self._field_key(i, "field", name, device, dtype)
                v = self.field_cache.get(k)
                if v is None:
                    fdict, _meta = self.store.read_sample(
                        i, fields=[name], coords=[], device=device, dtype=dtype, sample_ctor=lambda f, c, m: (f, m)
                    )
                    if name in fdict:
                        v = fdict[name]
                        self.field_cache.put(k, v)
                if v is not None:
                    out_fields[name] = v

            for name in req_coords:
                k = self._field_key(i, "coord", name, device, dtype)
                v = self.field_cache.get(k)
                if v is None:
                    cdict, _meta = self.store.read_sample(
                        i, fields=[], coords=[name], device=device, dtype=dtype, sample_ctor=lambda f, c, m: (c, m)
                    )
                    if name in cdict:
                        v = cdict[name]
                        self.field_cache.put(k, v)
                if v is not None:
                    out_coords[name] = v

            meta = self.store.meta(i)
            if sample_ctor is None:
                try:
                    from .physical_sample import PhysicalSample
                    sample = PhysicalSample(fields=out_fields, coords=out_coords, meta=meta)
                except Exception:
                    sample = {"fields": out_fields, "coords": out_coords, "meta": meta}
            else:
                sample = sample_ctor(out_fields, out_coords, meta)

            if use_sample_cache:
                self.sample_cache.put(sk, sample)
            return sample

        sample = self.store.read_sample(i, fields=fields, coords=coords, device=device, dtype=dtype, sample_ctor=sample_ctor)
        if use_sample_cache:
            self.sample_cache.put(sk, sample)
        return sample

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss/eviction and byte-usage statistics."""
        return {
            "sample_cache": {
                "items": len(self.sample_cache),
                "hits": self.sample_cache.stats.hits,
                "misses": self.sample_cache.stats.misses,
                "evictions": self.sample_cache.stats.evictions,
                "bytes": self.sample_cache.stats.bytes_in_use,
                "max_bytes": self.sample_cache.max_bytes,
            },
            "field_cache": {
                "items": len(self.field_cache),
                "hits": self.field_cache.stats.hits,
                "misses": self.field_cache.stats.misses,
                "evictions": self.field_cache.stats.evictions,
                "bytes": self.field_cache.stats.bytes_in_use,
                "max_bytes": self.field_cache.max_bytes,
            },
        }
