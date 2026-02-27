"""UPDDataset -> PINNFactory (data-only) end-to-end on a synthetic Zarr shard.

This example showcases the "I/O" side of pinneaple_pinn:
  - build a tiny xarray Dataset with coords (time, lat, lon)
  - write to Zarr + JSON meta, wrap as UPDItem
  - use PINNMapping + UPDDataset to sample batches
  - train a neural surrogate with PINNFactory using the "data" loss bucket

It's intentionally data-only (no PDE residuals) to keep it lightweight.
Once your UPD pipeline is in place, you can add PDE residuals + conditions
via SchemaAdapter or directly in PINNProblemSpec.

Run:
  python examples/pinneaple_pinn/05_upd_dataset_data_only_regression.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch
import xarray as xr

from pinneaple_pinn.factory.pinn_factory import NeuralNetwork, PINN, PINNFactory, PINNProblemSpec
from pinneaple_pinn.io import UPDItem, UPDDataset, build_default_mapping_atmosphere, SamplingSpec


def make_synthetic_atmosphere_ds() -> xr.Dataset:
    """Tiny gridded dataset with a smooth signal."""
    time = np.array(["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[D]")
    lat = np.linspace(-10.0, 10.0, 32, dtype=np.float32)
    lon = np.linspace(30.0, 50.0, 48, dtype=np.float32)

    # Signal: T(time,lat,lon) = sin(lat)*cos(lon) * exp(-t)
    # (lat/lon in degrees here; mapping can convert to radians and normalize)
    tt = np.arange(time.shape[0], dtype=np.float32)[:, None, None]
    lat2 = lat[None, :, None]
    lon2 = lon[None, None, :]
    T = (np.sin(np.deg2rad(lat2)) * np.cos(np.deg2rad(lon2)) * np.exp(-0.7 * tt)).astype(np.float32)

    ds = xr.Dataset(
        {
            "T": ("time", "lat", "lon", T),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"name": "synthetic_demo"},
    )
    return ds


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # --- Create a temporary UPD shard (Zarr + JSON meta) ---
    with tempfile.TemporaryDirectory() as tmp:
        zarr_path = os.path.join(tmp, "shard.zarr")
        meta_path = os.path.join(tmp, "meta.json")

        ds = make_synthetic_atmosphere_ds()
        ds.to_zarr(zarr_path, mode="w")
        json.dump({"kind": "synthetic", "vars": ["T"]}, open(meta_path, "w", encoding="utf-8"), indent=2)

        item = UPDItem(zarr_path=zarr_path, meta_path=meta_path)

        # Map coords -> independent vars and UPD vars -> dependent vars
        mapping = build_default_mapping_atmosphere(dep_vars=["T"], use_lev=False)
        upd = UPDDataset(item=item, mapping=mapping, device=device, dtype=dtype)

        # Data-only PINNProblemSpec (acts like a continuous regressor)
        spec = PINNProblemSpec(
            pde_residuals=[],
            conditions=[],
            independent_vars=mapping.independent_vars(),
            dependent_vars=mapping.dependent_vars(),
            inverse_params=[],
            loss_weights={"data": 1.0},
            verbose=False,
        )

        factory = PINNFactory(spec)
        loss_fn = factory.generate_loss_function()

        net = NeuralNetwork(num_inputs=len(spec.independent_vars), num_outputs=len(spec.dependent_vars), num_layers=4, num_neurons=64)
        model = PINN(net).to(device=device, dtype=dtype)

        opt = torch.optim.Adam(model.parameters(), lr=2e-3)

        # Sample supervised points from the shard
        sampling = SamplingSpec(n_collocation=0, n_data=4096, conditions=[], replace=True, seed=123)
        batch = upd.sample(sampling)
        assert batch.data is not None

        train_batch = {"data": batch.data}

        # --- Train ---
        steps = 1000
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, comps = loss_fn(model, train_batch)
            loss.backward()
            opt.step()
            if step % 200 == 0 or step == 1:
                print(f"step={step:04d} data_mse={comps.get('data', 0.0):.4e} total={comps['total']:.4e}")

        # quick eval on a second random sample
        sampling2 = SamplingSpec(n_collocation=0, n_data=1024, conditions=[], replace=True, seed=999)
        batch2 = upd.sample(sampling2)
        assert batch2.data is not None
        inputs2, y2 = batch2.data
        with torch.no_grad():
            pred2 = model(*inputs2).detach().cpu().numpy()
        y2n = y2.detach().cpu().numpy()
        rel = float(np.linalg.norm(pred2 - y2n) / (np.linalg.norm(y2n) + 1e-12))
        print(f"rel_L2(random_sample) = {rel:.3e}")


if __name__ == "__main__":
    main()
