"""30_zarr_data_pipeline.py — UPD + Zarr sharded data pipeline.

Demonstrates:
- UnifiedPhysicalData (UPD): standard container for physical samples
- ZarrDatasetWriter: write UPD samples to a sharded Zarr store
- ZarrDatasetReader: lazy-load with byte-based LRU cache and prefetch
- CollocationSampler: sample boundary + interior points from a domain
- DataValidator: physics-aware quality checks
"""

import os
import math
import torch
import numpy as np
import tempfile
import shutil

from pinneaple_data.upd import UnifiedPhysicalData, UPDSchema
from pinneaple_data.zarr_store import ZarrDatasetWriter, ZarrDatasetReader
from pinneaple_data.collocation import CollocationSampler, CollocationConfig
from pinneaple_data.validation import DataValidator, ValidationConfig
from pinneaple_geom.csg import CSGRectangle


# ---------------------------------------------------------------------------
# Scenario: generate a dataset of 2D heat conduction solutions for
# varying boundary temperatures and write them to a Zarr store.
# Each sample has:
#   inputs:  x (N,2) collocation points
#   outputs: T (N,1) temperature values from an analytic solution
#   params:  T_top, T_bottom (boundary temperatures)
# ---------------------------------------------------------------------------

NX = 32    # spatial resolution per sample
NY = 32
N_SAMPLES = 500


def analytic_temp(xy: np.ndarray, T_top: float, T_bot: float) -> np.ndarray:
    """Linear temperature profile T(x,y) = T_bot + (T_top - T_bot) * y."""
    return (T_bot + (T_top - T_bot) * xy[:, 1:2]).astype(np.float32)


def main():
    rng   = np.random.default_rng(42)
    store = tempfile.mkdtemp(prefix="pinneaple_zarr_")
    print(f"Zarr store at: {store}")

    # --- Geometry & collocation sampler ------------------------------------
    rect = CSGRectangle(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    col_config = CollocationConfig(
        n_interior=NX * NY,
        n_boundary=200,
        seed=0,
    )
    sampler = CollocationSampler(geometry=rect, config=col_config)

    # --- UPD schema --------------------------------------------------------
    schema = UPDSchema(
        input_fields=["xy_interior", "xy_boundary"],
        output_fields=["T_interior", "T_boundary"],
        param_fields=["T_top", "T_bot"],
    )

    # --- Write Zarr dataset ------------------------------------------------
    writer_config = {
        "store_path":  store,
        "chunk_size":  64,
        "compressor":  "lz4",
        "shard_size":  1024,
    }
    writer = ZarrDatasetWriter(schema=schema, **writer_config)
    writer.open()

    print(f"Writing {N_SAMPLES} UPD samples to Zarr ...")
    for i in range(N_SAMPLES):
        T_top = float(rng.uniform(50, 200))
        T_bot = float(rng.uniform(0, 50))

        xy_int = sampler.sample_interior(seed=i).astype(np.float32)
        xy_bnd = sampler.sample_boundary(seed=i + N_SAMPLES).astype(np.float32)
        T_int  = analytic_temp(xy_int, T_top, T_bot)
        T_bnd  = analytic_temp(xy_bnd, T_top, T_bot)

        sample = UnifiedPhysicalData(
            inputs={
                "xy_interior": xy_int,
                "xy_boundary": xy_bnd,
            },
            outputs={
                "T_interior": T_int,
                "T_boundary": T_bnd,
            },
            params={
                "T_top": np.array([T_top], dtype=np.float32),
                "T_bot": np.array([T_bot], dtype=np.float32),
            },
            metadata={"sample_id": i},
        )
        writer.write(sample)

    writer.close()
    print("Write complete.")

    # --- Read & validate ---------------------------------------------------
    reader = ZarrDatasetReader(
        store_path=store,
        schema=schema,
        cache_bytes=50 * 1024 * 1024,     # 50 MB LRU cache
        prefetch_factor=4,
    )
    reader.open()

    n_stored = reader.n_samples()
    print(f"\nZarr store: {n_stored} samples")

    # Data validation
    val_config = ValidationConfig(
        check_nans=True,
        check_infs=True,
        check_range={
            "T_interior": (0.0, 200.0),
            "T_boundary": (0.0, 200.0),
        },
        n_samples_to_check=50,
    )
    validator = DataValidator(reader=reader, config=val_config)
    val_report = validator.run()
    print("\nValidation report:")
    for key, ok in val_report.items():
        print(f"  {key:30s}: {'PASS' if ok else 'FAIL'}")

    # --- Benchmark read throughput -----------------------------------------
    import time
    print("\nBenchmarking random-access read speed ...")
    indices = rng.integers(0, n_stored, 200)
    t0 = time.perf_counter()
    for idx in indices:
        _ = reader.read(int(idx))
    elapsed = time.perf_counter() - t0
    print(f"  200 random reads: {elapsed:.3f} s  ({200/elapsed:.0f} samples/s)")

    # --- Example batch for training ----------------------------------------
    batch_size = 32
    batch_idx  = list(range(batch_size))
    batch      = [reader.read(i) for i in batch_idx]

    xy_batch = torch.tensor(
        np.stack([s.inputs["xy_interior"] for s in batch]), dtype=torch.float32
    )
    T_batch  = torch.tensor(
        np.stack([s.outputs["T_interior"] for s in batch]), dtype=torch.float32
    )
    print(f"\nSample batch: xy={tuple(xy_batch.shape)}  T={tuple(T_batch.shape)}")

    # Quick sanity check: L2 residual of linear profile
    T_linear = batch[0].params["T_bot"][0] + \
               (batch[0].params["T_top"][0] - batch[0].params["T_bot"][0]) * \
               batch[0].inputs["xy_interior"][:, 1]
    residual = np.abs(batch[0].outputs["T_interior"].ravel() - T_linear).max()
    print(f"  Analytic residual (sample 0): {residual:.6f}  (should be ~0)")

    reader.close()

    # Cleanup
    shutil.rmtree(store)
    print("\nZarr store cleaned up.")


if __name__ == "__main__":
    main()
