"""03_prefetch_streaming_gpu.py

Showcase: high-throughput streaming from a Zarr-backed UPD store with:
  - process-local Zarr caching
  - background prefetch thread
  - optional pinned-memory + async CPU->GPU transfer

Run:
  python examples/pinneaple_data/03_prefetch_streaming_gpu.py

Env knobs:
  TARGET_DEVICE=cpu|cuda
  NUM_WORKERS=0|1|2|4...
  N_TAKE=200
"""

from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_prefetch import PrefetchConfig, PrefetchZarrUPDIterable
from pinneaple_data.zarr_cached_store import ZarrCacheConfig


def _maybe_make_store(zarr_path: str, *, n_samples: int = 2000) -> None:
    if os.path.isdir(zarr_path):
        return

    os.makedirs(os.path.dirname(zarr_path) or ".", exist_ok=True)

    # A toy “UPD” sample: variable-length trajectories (T can vary), plus supervision.
    rng = torch.Generator().manual_seed(123)
    samples = []
    for i in range(n_samples):
        T = int(torch.randint(low=32, high=96, size=(1,), generator=rng).item())
        x = torch.randn(T, 4, generator=rng)
        y = torch.randn(T, 2, generator=rng)
        samples.append(
            PhysicalSample(
                state={"x": x, "y": y},
                domain={"type": "grid"},
                provenance={"i": i, "T": T},
            )
        )

    UPDZarrStore.write(
        zarr_path,
        samples,
        manifest={"name": "toy_prefetch", "n": n_samples, "note": "variable-length demo"},
    )


def main() -> None:
    out_dir = "examples/_out"
    zarr_path = os.path.join(out_dir, "toy_prefetch_ds.zarr")
    _maybe_make_store(zarr_path)

    target_device = os.environ.get("TARGET_DEVICE", "cpu")  # cpu|cuda
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    n_take = int(os.environ.get("N_TAKE", "200"))

    ds = PrefetchZarrUPDIterable(
        zarr_path,
        fields=["x", "y"],
        coords=[],
        cache=ZarrCacheConfig(max_bytes=256 * 1024 * 1024, max_items=10_000),
        prefetch_cfg=PrefetchConfig(
            prefetch=16,
            queue_max=64,
            use_sample_cache=True,
            pin_memory=(target_device.startswith("cuda")),
            target_device=target_device,
            transfer_non_blocking=True,
        ),
    )

    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    t0 = time.perf_counter()
    n = 0
    total_elems = 0
    for s in dl:
        x = s.state["x"]
        y = s.state["y"]

        if target_device.startswith("cuda"):
            assert x.is_cuda and y.is_cuda

        total_elems += int(x.numel() + y.numel())
        n += 1
        if n >= n_take:
            break

    dt = time.perf_counter() - t0
    ex_per_s = n / max(dt, 1e-9)
    melems_per_s = (total_elems / 1e6) / max(dt, 1e-9)
    print(
        "\n".join(
            [
                f"Zarr: {zarr_path}",
                f"device={target_device} workers={num_workers} n={n} dt={dt:.3f}s",
                f"throughput: {ex_per_s:.1f} samples/s | {melems_per_s:.1f} M elems/s",
            ]
        )
    )


if __name__ == "__main__":
    main()
