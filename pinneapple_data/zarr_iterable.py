"""IterableDataset for streaming PhysicalSamples from a UPDZarrStore with worker-aware sharding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .physical_sample import PhysicalSample
from .zarr_store import UPDZarrStore


@dataclass
class ZarrUPDIterable(IterableDataset):
    """
    IterableDataset over a UPDZarrStore.

    Key points:
      - Inherits from torch.utils.data.IterableDataset so DataLoader won't require __len__.
      - Opens the Zarr store inside __iter__ so each worker has its own handle.
      - Supports worker-aware sharding (each worker gets disjoint indices).
    """
    root: str
    fields: Sequence[str]
    coords: Sequence[str] = ()
    start: int = 0
    stop: Optional[int] = None
    step: int = 1
    shuffle: bool = False
    seed: int = 123

    def __iter__(self) -> Iterator[PhysicalSample]:
        """
        Iterate over samples from the store with optional shuffle and worker sharding.

        Yields
        ------
        PhysicalSample
            Sample from the Zarr store.
        """
        store = UPDZarrStore(self.root, mode="r")
        n = store.num_samples()

        start = int(self.start)
        stop = int(self.stop) if self.stop is not None else n
        step = int(self.step)
        if step <= 0:
            step = 1

        start = max(0, min(start, n))
        stop = max(0, min(stop, n))

        idxs = np.arange(start, stop, step, dtype=np.int64)

        if self.shuffle:
            rng = np.random.default_rng(int(self.seed))
            rng.shuffle(idxs)

        # ---- worker-aware split (important for num_workers>0)
        info = get_worker_info()
        if info is not None:
            wid = info.id
            wnum = info.num_workers
            idxs = idxs[wid::wnum]

        for idx in idxs:
            yield store.read_sample(int(idx), fields=self.fields)
