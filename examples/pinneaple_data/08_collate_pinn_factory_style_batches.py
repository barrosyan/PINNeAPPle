"""08_collate_pinn_factory_style_batches.py

Showcase: collate_pinn_batches — a collate_fn for PINN-style mixed batches.

Many PINN workflows want a single batch that contains multiple components:
  - collocation points (physics residual)
  - boundary/initial conditions (multiple condition sets)
  - supervised data (x,y) pairs

pinneaple_data.collate.collate_pinn_batches merges a list of per-item dicts into a
single dict where tensors are concatenated along dim=0.

Run:
  python examples/pinneaple_data/08_collate_pinn_factory_style_batches.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from pinneaple_data.collate import collate_pinn_batches


@dataclass
class ToyPINNItem:
    """Toy item emitting the same structure a PINNFactory dataset would emit."""

    n_col: int
    n_bc: int
    n_ic: int
    n_data: int

    def to_dict(self) -> Dict[str, Any]:
        # collocation: (t,x)
        t = torch.rand(self.n_col, 1)
        x = torch.rand(self.n_col, 1)

        # conditions: list of tuples, e.g. [BC(t,x,u), IC(t,x,u)]
        bc_t = torch.rand(self.n_bc, 1)
        bc_x = torch.zeros(self.n_bc, 1)  # x=0 boundary
        bc_u = torch.zeros(self.n_bc, 1)

        ic_t = torch.zeros(self.n_ic, 1)  # t=0 initial
        ic_x = torch.rand(self.n_ic, 1)
        ic_u = torch.sin(ic_x)

        # supervised data: (t,x)->u
        data_t = torch.rand(self.n_data, 1)
        data_x = torch.rand(self.n_data, 1)
        data_u = torch.sin(data_x) * torch.exp(-data_t)

        return {
            "collocation": (t, x),
            "conditions": [(bc_t, bc_x, bc_u), (ic_t, ic_x, ic_u)],
            "data": ((data_t, data_x), data_u),
            "meta": {"n_col": self.n_col, "n_data": self.n_data},
        }


class ToyPINNDataset(Dataset):
    def __init__(self, n: int = 64):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Variable sizes per item to demonstrate concat behavior
        n_col = 64 + (idx % 3) * 32
        n_bc = 16
        n_ic = 16
        n_data = 8 + (idx % 5) * 4
        return ToyPINNItem(n_col=n_col, n_bc=n_bc, n_ic=n_ic, n_data=n_data).to_dict()


def main() -> None:
    ds = ToyPINNDataset(n=20)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pinn_batches)

    batch = next(iter(dl))
    (t_col, x_col) = batch["collocation"]
    conds: List[Tuple[torch.Tensor, ...]] = batch["conditions"]
    (data_tx, data_u) = batch["data"]
    metas = batch["meta"]

    print("collocation:", t_col.shape, x_col.shape)
    print("conditions:")
    for i, c in enumerate(conds):
        print(f"  cond[{i}]:", [tuple(t.shape) for t in c])
    print("data:", [tuple(t.shape) for t in data_tx], tuple(data_u.shape))
    print("meta count:", len(metas), "example:", metas[0])


if __name__ == "__main__":
    main()
