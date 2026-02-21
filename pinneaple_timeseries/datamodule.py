"""Time series DataModule with train/val split and loaders."""
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, random_split
from .datasets.windowed import WindowedTimeSeriesDataset
from .spec import TimeSeriesSpec


@dataclass
class TSDataModule:
    series: torch.Tensor
    spec: TimeSeriesSpec
    batch_size: int = 64
    val_ratio: float = 0.2

    def make_loaders(self):
        dataset = WindowedTimeSeriesDataset(self.series, self.spec)

        n = len(dataset)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val

        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val_ds, batch_size=max(128, self.batch_size))

        return train, val
