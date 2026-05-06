import numpy as np
import xarray as xr

from pinneaple_data.physical_sample import PhysicalSample

ds = xr.Dataset(
    data_vars=dict(
        T2M=(("t","x"), np.random.randn(24, 16).astype("float32")),
        U10M=(("t","x"), np.random.randn(24, 16).astype("float32")),
    ),
    coords=dict(t=np.arange(24), x=np.arange(16)),
)

sample = PhysicalSample(
    state=ds,
    domain={"type": "grid"},
    schema={"governing": {"type": "PDE", "name": "toy"}},
    provenance={"source": "demo"},
)

print("Summary:", sample.summary())
print("Vars:", sample.list_variables())

td = sample.to_train_dict(x_vars=["T2M", "U10M"], y_vars=None, coords=["t","x"], time_dim="t")
print("Train dict keys:", td.keys())
print("x shape:", td["x"].shape)
