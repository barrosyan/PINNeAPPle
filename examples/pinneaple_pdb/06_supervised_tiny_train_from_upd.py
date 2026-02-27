"""Tiny supervised training loop from a UPD shard (no PDE residual yet).

Goal:
  show a clean path:
    UPD shard (xarray zarr) -> PINNMapping -> sampled (t,lat,lon)->T pairs -> torch NN

This is a stepping stone to full PINN training (physics residual + conditions).

Run after:
  - 03_offline_synthetic_upd_pipeline.py

Requirements:
  pip install pandas pyarrow xarray zarr torch numpy
"""

from __future__ import annotations

import os
import pandas as pd
import torch

from pinneaple_pinn.io.upd_dataset import UPDItem, UPDDataset, SamplingSpec
from pinneaple_pinn.io.mappings import CoordMapping, VarMapping, PINNMapping


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    catalog_path = "examples/_out/pinneaple_pdb/synthetic_build/catalog.parquet"
    if not os.path.exists(catalog_path):
        raise FileNotFoundError("Run 03_offline_synthetic_upd_pipeline.py first.")

    df = pd.read_parquet(catalog_path).sort_values("uid")
    row = df.iloc[0].to_dict()
    item = UPDItem(zarr_path=row["zarr_path"], meta_path=row["meta_path"])

    mapping = PINNMapping(
        coord=CoordMapping(
            ind_vars=["t", "lat", "lon"],
            coord_sources={"t": "time", "lat": "lat", "lon": "lon"},
            coord_transform={"t": "seconds_since_start", "lat": "deg2rad", "lon": "deg2rad"},
            normalize_to_unit=True,
        ),
        var=VarMapping(dep_vars=["T"], var_sources={"T": "T2M"}, normalize_to_unit=True),
    )

    ds = UPDDataset(item=item, mapping=mapping, device="cpu", dtype=torch.float32)

    model = MLP(d_in=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        batch = ds.sample(SamplingSpec(n_collocation=0, n_data=2048, seed=step))
        xin, y = batch.data
        # xin is tuple(t,lat,lon) each (N,1)
        x = torch.cat(list(xin), dim=1)     # (N,3)
        y = y                               # (N,1)

        yhat = model(x)
        loss = torch.mean((yhat - y) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0:
            print(f"step={step:04d} loss={loss.item():.6f}")

    print("done.")


if __name__ == "__main__":
    main()
