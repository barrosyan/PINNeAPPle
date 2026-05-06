import os
import torch
from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_iterable import ZarrUPDIterable

out_dir = "examples/_out"
os.makedirs(out_dir, exist_ok=True)
zarr_path = os.path.join(out_dir, "toy_ds.zarr")

if not os.path.isdir(zarr_path):
    samples = []
    for i in range(200):
        x = torch.randn(64, 4)   # (T,D)
        y = torch.randn(64, 2)
        samples.append(PhysicalSample(state={"x": x, "y": y}, domain={"type":"grid"}, provenance={"i": i}))
    UPDZarrStore.write(zarr_path, samples, manifest={"name":"toy"})

ds = ZarrUPDIterable(zarr_path, fields=["x","y"], coords=[])
dl = DataLoader(ds, batch_size=None, num_workers=2, persistent_workers=True)

for i, s in enumerate(dl):
    x = s.state["x"]
    y = s.state["y"]
    print(i, x.shape, y.shape)
    if i >= 3:
        break

