"""DataModule, loaders, and item adapters for training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .splits import SplitSpec, split_indices
from .preprocess import PreprocessPipeline


# ----------------------------
# Adapters / simple datasets
# ----------------------------
class ItemAdapter:
    """
    Adapter that standardizes how we convert dataset items into dict batches.

    Output dict must have at least:
      - "x": Tensor
      - optional "y": Tensor
      - optional "meta": dict
      - optional "physics": dict (for PINN hooks)
    """
    def __call__(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return {"x": item[0], "y": item[1]}
        raise TypeError("ItemAdapter expects dict or (x,y) tuple/list")


class FnAdapter(ItemAdapter):
    """
    Convenience adapter for Pinneaple PhysicalSample-like objects (or any object).

    You pass:
      x_fn(sample) -> Tensor
      y_fn(sample) -> Tensor
      meta_fn(sample) -> dict (optional)
    """
    def __init__(
        self,
        x_fn: Callable[[Any], torch.Tensor],
        y_fn: Callable[[Any], torch.Tensor],
        meta_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ):
        self.x_fn = x_fn
        self.y_fn = y_fn
        self.meta_fn = meta_fn or (lambda s: {})

    def __call__(self, sample: Any) -> Dict[str, Any]:
        return {"x": self.x_fn(sample), "y": self.y_fn(sample), "meta": self.meta_fn(sample)}


class AdaptedSequenceDataset(Dataset):
    """
    Wrap a Python sequence and apply an adapter in __getitem__.
    Useful for: list[PhysicalSample] or list[dict] etc.
    """
    def __init__(self, samples: Sequence[Any], adapter: ItemAdapter):
        self.samples = list(samples)
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.adapter(self.samples[idx])


# ----------------------------
# DataModule
# ----------------------------
@dataclass
class DataModule:
    """
    Builds train/val/test DataLoaders with consistent split + preprocess.

    Notes:
      - Uses split_indices(...) from splits.py (no make_splits required).
      - split default uses default_factory (Python 3.13 dataclasses).
      - Caches split subsets after first setup().
    """
    dataset: Union[Dataset, Sequence[Any]]
    split: SplitSpec = field(default_factory=SplitSpec)

    # if you want leakage-free splits by group
    group_ids: Optional[Sequence[str]] = None

    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False

    adapter: ItemAdapter = field(default_factory=ItemAdapter)
    preprocess: Optional[PreprocessPipeline] = None

    _cached: Optional[Tuple[Dataset, Dataset, Dataset]] = field(default=None, init=False, repr=False)

    def _as_dataset(self) -> Dataset:
        # If user passed a raw sequence, wrap it into a Dataset.
        if isinstance(self.dataset, Dataset):
            return self.dataset
        return AdaptedSequenceDataset(self.dataset, adapter=self.adapter)

    def setup(self) -> Tuple[Dataset, Dataset, Dataset]:
        if self._cached is not None:
            return self._cached

        base_ds = self._as_dataset()

        # Must have __len__ for index-based splitting (Subset)
        if not hasattr(base_ds, "__len__"):
            raise TypeError(
                "DataModule requires a sized Dataset (implements __len__) for splitting. "
                "For streaming datasets, use a different DataModule that splits by shards."
            )

        n = len(base_ds)
        idx = split_indices(n, self.split, group_ids=self.group_ids)

        train_ds = Subset(base_ds, idx["train"].tolist())
        val_ds = Subset(base_ds, idx["val"].tolist())
        test_ds = Subset(base_ds, idx["test"].tolist())

        self._cached = (train_ds, val_ds, test_ds)
        return self._cached

    def _collate(self, batch):
        """
        Collate list[dict] -> dict of tensors (stacked) + lists for non-tensors.
        """
        # If dataset already returns dict, keep it; otherwise adapt here (Subset wraps original ds)
        items = []
        for b in batch:
            if isinstance(b, dict):
                items.append(b)
            else:
                items.append(self.adapter(b))

        out: Dict[str, Any] = {}
        keys = set().union(*[it.keys() for it in items])
        for k in keys:
            vals = [it.get(k) for it in items]
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out

    def train_dataloader(self) -> DataLoader:
        train_ds, _, _ = self.setup()
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(self.split.method != "time"),  # time split usually shouldn't shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=self.drop_last,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        _, val_ds, _ = self.setup()
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        _, _, test_ds = self.setup()
        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
            collate_fn=self._collate,
        )
