"""06_sharded_zarr_balanced_workers.py

Showcase: sharded UPD Zarr layout for "industrial" scale datasets.

Why sharding matters:
  - Partition by regime/time/region so you can download/process subsets.
  - Each shard is a self-contained Zarr store.
  - DataLoader workers are balanced across shards automatically.

This example:
  1) Generates a mixed synthetic dataset (heat/advection/logistic)
  2) Writes shards by governing-equation kind
  3) Streams back with ShardAwareZarrUPDIterable using multiple workers

Run:
  python examples/pinneaple_data/06_sharded_zarr_balanced_workers.py

Env knobs:
  TARGET_DEVICE=cpu|cuda
  NUM_WORKERS=4
"""

from __future__ import annotations

import os
from collections import Counter

from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.synth.pde import PDESynthGenerator
from pinneaple_data.synth.sample_adapter import to_physical_sample
from pinneaple_data.zarr_shards import ShardSpec, UPDZarrShardedWriter
from pinneaple_data.zarr_shard_iterable import ShardAwareConfig, ShardAwareZarrUPDIterable
from pinneaple_data.zarr_cached_store_bytes import ZarrByteCacheConfig


def _as_ps(sample_like: object, *, kind: str) -> PhysicalSample:
    ps = to_physical_sample(sample_like)
    if isinstance(ps, PhysicalSample):
        ps.domain = {"type": "grid"}
        ps.schema = {"governing": {"type": "PDE/ODE", "name": kind}}
        ps.provenance = {"source": "pinneaple_data.synth", "kind": kind}
        return ps
    return PhysicalSample(
        state={"fields": dict(ps.fields), "coords": dict(ps.coords)},  # type: ignore[attr-defined]
        domain={"type": "grid"},
        schema={"governing": {"type": "PDE/ODE", "name": kind}},
        provenance={"source": "pinneaple_data.synth", "kind": kind},
        extras={"meta": dict(ps.meta)},  # type: ignore[attr-defined]
    )


def main() -> None:
    out_dir = "examples/_out"
    os.makedirs(out_dir, exist_ok=True)
    root = os.path.join(out_dir, "synth_sharded")

    if not os.path.isdir(os.path.join(root, "shards")):
        gen = PDESynthGenerator(seed=11)
        samples = []
        specs = [
            ("heat1d", dict(T=64, X=64, alpha=0.03), 120),
            ("advection1d", dict(T=64, X=64, c=1.2), 80),
            ("logistic", dict(T=128, r=1.0, K=1.5), 60),
        ]

        for kind, cfg, n in specs:
            for _ in range(n):
                out = gen.sample(kind=kind, **cfg)
                samples.append(_as_ps(out.sample, kind=kind))

        writer = UPDZarrShardedWriter(
            root,
            ShardSpec(
                key_fn=lambda s: (s.provenance or {}).get("kind", "unknown"),
                max_per_shard=50,
            ),
        )
        writer.write(samples)
        print(f"Wrote sharded dataset -> {root}")

    target_device = os.environ.get("TARGET_DEVICE", "cpu")
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))

    ds = ShardAwareZarrUPDIterable(
        root,
        fields=None,
        coords=None,
        cache=ZarrByteCacheConfig(max_bytes=256 * 1024 * 1024, max_items=50_000),
        cfg=ShardAwareConfig(
            pin_memory=target_device.startswith("cuda"),
            target_device=target_device,
            transfer_non_blocking=True,
            use_sample_cache=True,
        ),
    )
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers, persistent_workers=True)

    kinds = []
    for i, s in enumerate(dl):
        kinds.append((s.provenance or {}).get("kind", "unknown"))
        if i >= 299:
            break

    print("Sample kind distribution (first 300 streamed):", dict(Counter(kinds)))


if __name__ == "__main__":
    main()
