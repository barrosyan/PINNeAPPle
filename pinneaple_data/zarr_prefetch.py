"""Prefetch IterableDataset for Zarr-backed UPD stores with background producer and optional GPU transfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, List
import threading
import queue

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .zarr_cached_store import CachedUPDZarrStore, ZarrCacheConfig
from .device import pin_sample, to_device_sample


@dataclass
class PrefetchConfig:
    """
    Industrial prefetch controls.

    pin_memory:
      - If True, producer pins CPU tensors (good for GPU training).
    target_device:
      - If "cuda", consumer transfers to GPU using non_blocking=True.
      - If "cpu", yields CPU tensors.
    transfer_non_blocking:
      - If True, uses non_blocking transfers (requires pinned memory for full benefit).
    queue_max:
      - buffer size between producer and consumer.
    """
    prefetch: int = 16
    queue_max: int = 32
    use_sample_cache: bool = True
    pin_memory: bool = True
    target_device: str = "cpu"  # "cpu" or "cuda"
    transfer_non_blocking: bool = True


class PrefetchZarrUPDIterable(IterableDataset):
    """
    Streaming dataset backed by Zarr with caching + background prefetch,
    plus pinned memory and async CPU->GPU transfer.

    Pattern:
      - Producer thread reads sample from Zarr into CPU tensors (+ optional pin)
      - Consumer yields either CPU or moved-to-GPU sample (non_blocking)

    Works with DataLoader(num_workers>0):
      - each worker has its own process-local cache and thread
    """
    def __init__(
        self,
        root: str,
        *,
        fields: Optional[Sequence[str]] = None,
        coords: Optional[Sequence[str]] = None,
        dtype: Optional[torch.dtype] = None,
        start: int = 0,
        end: Optional[int] = None,
        stride: int = 1,
        sample_ctor: Any = None,
        cache: Optional[ZarrCacheConfig] = None,
        prefetch_cfg: Optional[PrefetchConfig] = None,
    ):
        """
        Initialize the prefetch iterable dataset.

        Parameters
        ----------
        root : str
            Root path of the Zarr store.
        fields : Optional[Sequence[str]], optional
            Field names to load. Default is None (all fields).
        coords : Optional[Sequence[str]], optional
            Coordinate names to load. Default is None.
        dtype : Optional[torch.dtype], optional
            Target dtype for tensors.
        start : int, optional
            Start index. Default is 0.
        end : Optional[int], optional
            End index (exclusive). Default is None.
        stride : int, optional
            Step between indices. Default is 1.
        sample_ctor : Any, optional
            Optional sample constructor.
        cache : Optional[ZarrCacheConfig], optional
            Cache configuration.
        prefetch_cfg : Optional[PrefetchConfig], optional
            Prefetch configuration.
        """
        super().__init__()
        self.root = root
        self.fields = fields
        self.coords = coords
        self.dtype = dtype
        self.start = start
        self.end = end
        self.stride = stride
        self.sample_ctor = sample_ctor
        self.cache = cache or ZarrCacheConfig()
        self.prefetch_cfg = prefetch_cfg or PrefetchConfig()

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over samples with background prefetch, optional pinning, and GPU transfer.

        Yields
        ------
        Any
            Sample from the store, optionally moved to target device.
        """
        store = CachedUPDZarrStore(self.root, cache=self.cache, mode="r")
        n = store.count()
        end = n if self.end is None else min(self.end, n)

        wi = get_worker_info()
        if wi is None:
            indices = list(range(self.start, end, self.stride))
        else:
            worker_id = wi.id
            num_workers = wi.num_workers
            base = self.start + worker_id * self.stride
            step = num_workers * self.stride
            indices = list(range(base, end, step))

        q: "queue.Queue[Any]" = queue.Queue(maxsize=self.prefetch_cfg.queue_max)
        stop = threading.Event()
        err_holder: List[BaseException] = []

        # Always read in CPU first for Zarr IO
        io_device = "cpu"

        def producer():
            """Producer thread that reads samples from the store and enqueues them."""
            try:
                for i in indices:
                    if stop.is_set():
                        break
                    s = store.read_sample(
                        i,
                        fields=self.fields,
                        coords=self.coords,
                        device=io_device,
                        dtype=self.dtype,
                        sample_ctor=self.sample_ctor,
                        use_sample_cache=self.prefetch_cfg.use_sample_cache,
                    )
                    if self.prefetch_cfg.pin_memory:
                        s = pin_sample(s)
                    q.put(s)
                q.put(None)  # sentinel
            except BaseException as e:
                err_holder.append(e)
                q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()

        # Consumer side: optional CUDA transfer
        target = self.prefetch_cfg.target_device.lower()
        do_cuda = target.startswith("cuda")

        while True:
            item = q.get()
            if item is None:
                break
            if err_holder:
                raise err_holder[0]

            if do_cuda:
                item = to_device_sample(
                    item,
                    device="cuda",
                    dtype=self.dtype,
                    non_blocking=self.prefetch_cfg.transfer_non_blocking,
                )

            yield item

        stop.set()
