from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class SingleBatchDataset(Dataset):
    """
    Wraps a prebuilt dict-batch (e.g. from STLDomainBatchBuilder) into a Dataset.

    It yields samples by indexing:
      - x_col is used as primary length
      - x_bc / y_bc are cycled
      - x_data / y_data are cycled
      - x_ic / y_ic are cycled
    """
    def __init__(self, batch: Dict[str, Any]):
        self.batch = batch
        self.N = int(batch["x_col"].shape[0])

        def _len(key: str) -> int:
            t = batch.get(key)
            if t is None:
                return 0
            return int(t.shape[0])

        self.N_bc = _len("x_bc")
        self.N_data = _len("x_data")
        self.N_ic = _len("x_ic")

    def __len__(self):
        return self.N

    def __getitem__(self, i: int):
        out = {
            "x_col": self.batch["x_col"][i],
        }

        if self.N_bc > 0:
            j = i % self.N_bc
            out["x_bc"] = self.batch["x_bc"][j]
            out["y_bc"] = self.batch["y_bc"][j]
            if "n_bc" in self.batch:
                out["n_bc"] = self.batch["n_bc"][j]

            # masks are full-length over x_bc; keep them in batch-level ctx
            for k in list(self.batch.keys()):
                if k.startswith("mask_"):
                    out[k] = self.batch[k][j]

        if self.N_data > 0:
            j = i % self.N_data
            out["x_data"] = self.batch["x_data"][j]
            out["y_data"] = self.batch["y_data"][j]

        if self.N_ic > 0:
            j = i % self.N_ic
            out["x_ic"] = self.batch["x_ic"][j]
            out["y_ic"] = self.batch["y_ic"][j]

        # ctx is constant
        out["ctx"] = self.batch.get("ctx", {})
        return out


def dict_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            # keep ctx as-is (first)
            out[k] = v0
    return out