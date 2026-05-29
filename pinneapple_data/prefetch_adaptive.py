"""Adaptive prefetch IterableDataset for Zarr-backed UPD stores with configurable queue control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, List
import threading
import queue
import time

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .device import pin_sample, to_device_sample
from .zarr_cached_store_bytes import CachedUPDZarrStoreBytes, ZarrByteCacheConfig


@dataclass
class AdaptivePrefetchConfig:
    """
    Configuration for adaptive prefetching behavior.

    This config controls:
      - The maximum queue capacity used for prefetch buffering.
      - An adaptive target fill level (within bounds) used by a producer thread
        to regulate how many prefetched samples to keep buffered.
      - A control loop that adjusts the target fill based on queue occupancy.
      - Optional pinning and device transfer behaviors.

    Attributes
    ----------
    queue_max : int
        Hard maximum capacity of the prefetch queue.
    min_target_fill : int
        Minimum allowed adaptive target fill level.
    max_target_fill : int
        Maximum allowed adaptive target fill level.
    target_fill_init : int
        Initial target fill level used before the controller adapts it.
    control_period_s : float
        Period (seconds) for the controller to evaluate queue occupancy.
    increase_step : int
        Step size used to increase the target fill when queue is frequently drained.
    decrease_step : int
        Step size used to decrease the target fill when queue stays near full.
    high_watermark : float
        Occupancy ratio threshold (qsize / queue_max) above which target fill decreases.
    low_watermark : float
        Occupancy ratio threshold below which target fill increases.
    use_sample_cache : bool
        Whether to enable per-sample caching at the store level.
    pin_memory : bool
        Whether to pin sample tensors in CPU pinned memory after reading.
    target_device : str
        Desired device for yielded samples ("cpu" or "cuda...").
    transfer_non_blocking : bool
        Whether GPU transfers should be non-blocking when supported.
    """
    # Hard queue capacity (cannot change at runtime)
    queue_max: int = 64

    # Adaptive target fill bounds (producer tries to keep queue around target_fill)
    min_target_fill: int = 4
    max_target_fill: int = 32
    target_fill_init: int = 16

    # Control loop
    control_period_s: float = 0.5
    increase_step: int = 2
    decrease_step: int = 2

    # If consumer is slow (queue stays near full), decrease. If queue drains often, increase.
    high_watermark: float = 0.85   # qsize/queue_max
    low_watermark: float = 0.25

    # Caching + pin + device
    use_sample_cache: bool = True
    pin_memory: bool = True
    target_device: str = "cpu"     # "cpu" or "cuda"
    transfer_non_blocking: bool = True


class AdaptivePrefetchZarrUPDIterable(IterableDataset):
    """
    IterableDataset that streams samples from a cached Zarr-backed UPD store
    with adaptive prefetching.

    Prefetching model
    -----------------
    - A bounded queue buffers prefetched samples (hard capacity = cfg.queue_max).
    - A producer thread reads samples and pushes them to the queue.
    - A controller thread periodically adjusts a shared `target_fill` value
      within [cfg.min_target_fill, cfg.max_target_fill], based on queue occupancy:
        * If occupancy > cfg.high_watermark: decrease target fill.
        * If occupancy < cfg.low_watermark: increase target fill.
    - The producer throttles itself to keep the queue size below `target_fill`.

    Multi-worker behavior
    ---------------------
    When used with a PyTorch DataLoader with multiple workers, each worker process:
    - Builds its own store instance and its own producer/controller threads.
    - Receives a disjoint subset of indices based on worker_id and stride.

    Device behavior
    ---------------
    - Samples are read on CPU.
    - If cfg.pin_memory is True, tensor fields/coords are pinned.
    - If cfg.target_device starts with "cuda", samples are moved to GPU before yield.
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
        cache: Optional[ZarrByteCacheConfig] = None,
        cfg: Optional[AdaptivePrefetchConfig] = None,
    ):
        """
        Initialize the iterable dataset.

        Parameters
        ----------
        root : str
            Root path or identifier for the underlying Zarr UPD store.
        fields : Optional[Sequence[str]]
            Optional subset of field variables to load per sample.
        coords : Optional[Sequence[str]]
            Optional subset of coordinate variables to load per sample.
        dtype : Optional[torch.dtype]
            Optional dtype to request when reading samples.
        start : int
            Starting sample index (inclusive).
        end : Optional[int]
            Ending sample index (exclusive). If None, uses store.count().
        stride : int
            Step between consecutive indices.
        sample_ctor : Any
            Optional constructor or factory used by the store to build sample objects.
        cache : Optional[ZarrByteCacheConfig]
            Cache configuration for Zarr byte caching. Defaults to ZarrByteCacheConfig().
        cfg : Optional[AdaptivePrefetchConfig]
            Adaptive prefetch configuration. Defaults to AdaptivePrefetchConfig().
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
        self.cache = cache or ZarrByteCacheConfig()
        self.cfg = cfg or AdaptivePrefetchConfig()

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over samples, using adaptive prefetch buffering.

        Yields
        ------
        Any
            Sample object returned by the underlying store, optionally pinned
            and/or transferred to GPU according to the configuration.

        Raises
        ------
        BaseException
            Re-raises the first exception encountered in the producer thread.
        """
        store = CachedUPDZarrStoreBytes(self.root, cache=self.cache, mode="r")
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

        q: "queue.Queue[Any]" = queue.Queue(maxsize=self.cfg.queue_max)
        stop = threading.Event()
        err_holder: List[BaseException] = []

        # Adaptive state (shared within worker process)
        target_fill = {"v": int(self.cfg.target_fill_init)}
        target_lock = threading.Lock()

        def clamp(x: int) -> int:
            """
            Clamp a target fill value to the configured min/max bounds.
            """
            return max(self.cfg.min_target_fill, min(self.cfg.max_target_fill, x))

        def controller():
            """
            Control loop that periodically adjusts target_fill based on queue occupancy.

            The controller runs until `stop` is set, sleeping for cfg.control_period_s
            between updates.
            """
            while not stop.is_set():
                time.sleep(self.cfg.control_period_s)
                occ = q.qsize() / float(self.cfg.queue_max)
                with target_lock:
                    cur = target_fill["v"]
                    if occ > self.cfg.high_watermark:
                        target_fill["v"] = clamp(cur - self.cfg.decrease_step)
                    elif occ < self.cfg.low_watermark:
                        target_fill["v"] = clamp(cur + self.cfg.increase_step)

        io_device = "cpu"

        def producer():
            """
            Producer loop that reads samples from the store and enqueues them.

            The producer throttles itself to keep queue size below the current
            adaptive target_fill, pins samples if configured, and appends any
            encountered exception to err_holder before terminating iteration.
            """
            try:
                for i in indices:
                    if stop.is_set():
                        break

                    # wait until queue is below current target_fill
                    while not stop.is_set():
                        with target_lock:
                            tf = target_fill["v"]
                        if q.qsize() < tf:
                            break
                        time.sleep(0.001)

                    s = store.read_sample(
                        i,
                        fields=self.fields,
                        coords=self.coords,
                        device=io_device,
                        dtype=self.dtype,
                        sample_ctor=self.sample_ctor,
                        use_sample_cache=self.cfg.use_sample_cache,
                    )
                    if self.cfg.pin_memory:
                        s = pin_sample(s)
                    q.put(s)

                q.put(None)
            except BaseException as e:
                err_holder.append(e)
                q.put(None)

        th_prod = threading.Thread(target=producer, daemon=True)
        th_ctrl = threading.Thread(target=controller, daemon=True)

        th_prod.start()
        th_ctrl.start()

        do_cuda = self.cfg.target_device.lower().startswith("cuda")

        while True:
            item = q.get()
            if item is None:
                break
            if err_holder:
                raise err_holder[0]
            if do_cuda:
                item = to_device_sample(item, device="cuda", dtype=self.dtype, non_blocking=self.cfg.transfer_non_blocking)
            yield item

        stop.set()