from __future__ import annotations

import argparse
import os
import time
import json
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader

# Allow running as `python benchmarks/shard_balance_bench.py` or `python -m benchmarks.shard_balance_bench`
try:
    from ._latency import latency_summary  # type: ignore
except Exception:  # pragma: no cover
    from benchmarks._latency import latency_summary  # type: ignore


def _require_pinnea_data():
    """Import pinneaple_data with a helpful error if optional deps are missing."""
    try:
        from pinneaple_data.physical_sample import PhysicalSample
        from pinneaple_data.zarr_shards import UPDZarrShardedWriter, ShardSpec
        from pinneaple_data.zarr_shard_iterable import ShardAwareZarrUPDIterable, ShardAwareConfig
        from pinneaple_data.zarr_cached_store_bytes import ZarrByteCacheConfig

        return (
            PhysicalSample,
            UPDZarrShardedWriter,
            ShardSpec,
            ShardAwareZarrUPDIterable,
            ShardAwareConfig,
            ZarrByteCacheConfig,
        )
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise SystemExit(
            "\n".join(
                [
                    "[ERROR] Missing dependency while importing pinneaple_data.",
                    f"Missing module: {missing}",
                    "",
                    "Fix:",
                    "  pip install -e .",
                    "(or install the missing package directly, e.g. `pip install zarr`)",
                ]
            )
        ) from e


def make_sharded(root: str, n: int, shards: int, T: int, X: int) -> None:
    PhysicalSample, UPDZarrShardedWriter, ShardSpec, *_ = _require_pinnea_data()

    index_path = os.path.join(root, "index.json")
    if os.path.exists(index_path):
        return

    os.makedirs(root, exist_ok=True)

    samples = []
    per = max(1, n // max(1, shards))

    for i in range(n):
        shard_id = min(max(1, shards) - 1, i // per)
        u = torch.randn(T, X)

        # NEW PhysicalSample layout:
        # - state: Dict[str, Any] or xr.Dataset
        # - provenance: lineage/ids/time key, etc.
        # - domain: indicates interpretation of state (grid/mesh/graph)
        samples.append(
            PhysicalSample(
                state={"u": u},
                domain={"type": "grid"},
                provenance={
                    "time_key": f"2020-{shard_id+1:02d}",
                    "sample_id": f"s{i:06d}",
                },
            )
        )

    writer = UPDZarrShardedWriter(
        root,
        shard_spec=ShardSpec(
            # NEW: shard key comes from provenance now (instead of meta)
            key_fn=lambda s: f"time={getattr(s, 'provenance', {}).get('time_key', 'unknown')}",
            max_per_shard=per,
        ),
    )
    writer.write(samples)
    print(f"[OK] Wrote shards at {root} (n={n}, shards≈{shards}, per≈{per})")


def _get_u_from_sample(sample: Any) -> torch.Tensor:
    """
    Robustly extract 'u' tensor from the yielded sample.

    Your ShardAware pipeline yields a PhysicalSample by default (from CachedUPDZarrStoreBytes),
    where tensors live in sample.state (dict) when state is not xr.Dataset.
    """
    # PhysicalSample path
    if hasattr(sample, "state"):
        st = sample.state
        if isinstance(st, dict) and "u" in st:
            return st["u"]
        # If in the future you use xr.Dataset, you can handle it here too:
        # if isinstance(st, xr.Dataset): return torch.as_tensor(st["u"].data)
    # Fallback legacy dict-like
    if isinstance(sample, dict):
        # common legacy shapes
        if "fields" in sample and isinstance(sample["fields"], dict) and "u" in sample["fields"]:
            return sample["fields"]["u"]
        if "state" in sample and isinstance(sample["state"], dict) and "u" in sample["state"]:
            return sample["state"]["u"]

    raise KeyError("Could not find tensor 'u' inside sample (expected sample.state['u']).")


def run_iter(root: str, steps: int, workers: int, device: str, latency_max: int) -> Dict[str, Any]:
    (
        PhysicalSample,
        UPDZarrShardedWriter,
        ShardSpec,
        ShardAwareZarrUPDIterable,
        ShardAwareConfig,
        ZarrByteCacheConfig,
    ) = _require_pinnea_data()

    cfg = ShardAwareConfig(
        pin_memory=True,
        target_device=("cuda" if device == "cuda" else "cpu"),
        transfer_non_blocking=True,
        use_sample_cache=True,
    )

    ds = ShardAwareZarrUPDIterable(
        root,
        fields=["u"],     # these become sample.state keys
        coords=[],
        cfg=cfg,
        cache=ZarrByteCacheConfig(
            max_sample_bytes=256 * 1024 * 1024,
            max_field_bytes=256 * 1024 * 1024,
            enable_field_cache=True,
        ),
    )

    dl = DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=(workers > 0))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    lat: List[float] = []
    t0 = time.perf_counter()
    n = 0
    checksum = 0.0

    it = iter(dl)
    while n < steps:
        t1 = time.perf_counter()
        s = next(it)

        u = _get_u_from_sample(s)
        checksum += float(u.mean().item())

        t2 = time.perf_counter()
        if len(lat) < latency_max:
            lat.append((t2 - t1) * 1000.0)
        n += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    return {
        "steps": n,
        "seconds": dt,
        "samples_per_s": n / max(dt, 1e-9),
        "latency": latency_summary(lat),
        "cuda_peak_mb": peak,
        "checksum": checksum,
        "workers": workers,
        "device": device,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bench_out/shards")
    ap.add_argument("--root", type=str, default="bench_out/shards/sharded_root")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--X", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--latency_max", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    make_sharded(args.root, args.n, args.shards, args.T, args.X)

    r = run_iter(args.root, args.steps, args.workers, args.device, args.latency_max)

    out_json = os.path.join(args.out, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "result": r, "torch": torch.__version__}, f, indent=2)

    lat = r.get("latency", {})
    print(f"[OK] Wrote {out_json}")
    print(
        f"Shard-aware: {r['samples_per_s']:.1f} samples/s | "
        f"p50={lat.get('p50_ms', float('nan')):.2f}ms p95={lat.get('p95_ms', float('nan')):.2f}ms"
    )


if __name__ == "__main__":
    main()