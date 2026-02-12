import torch
from torch.utils.data import Dataset
from ..spec import TimeSeriesSpec


class WindowedTimeSeriesDataset(Dataset):
    """
    series: (T, F)
    x: (L_in, F)
    y: (H, C)   (aqui C = F por padrão)
    """
    def __init__(self, series: torch.Tensor, spec: TimeSeriesSpec):
        assert series.ndim == 2
        self.series = series.float()
        self.spec = spec

        T, F = self.series.shape
        L = spec.input_len
        H = spec.horizon
        off = spec.target_offset

        self.max_i = T - (L + off + H)
        if self.max_i < 0:
            raise ValueError("Série curta demais para input_len/horizon.")

        self.indices = list(range(0, self.max_i + 1, spec.stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        L = self.spec.input_len
        H = self.spec.horizon
        off = self.spec.target_offset

        x = self.series[i : i + L]                    # (L, F)
        y = self.series[i + L + off : i + L + off + H]  # (H, F)
        return x, y
