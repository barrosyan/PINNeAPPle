from __future__ import annotations

import argparse
import os
import time
import json
from dataclasses import asdict
from typing import Dict, Any, List

import torch

# Allow running as `python benchmarks/data_io_bench.py` or `python -m benchmarks.data_io_bench`
try:
    from ._latency import latency_summary  # type: ignore
except Exception:  # pragma: no cover
    from benchmarks._latency import latency_summary  # type: ignore


def _require_pinnea_data():
    """Import pinneaple_data with a helpful error if optional deps are missing."""
    try:
        from pinneaple_data import (  # noqa: WPS433 (runtime import)
            PhysicalSample,
            UPDZarrStore,
            PrefetchZarrUPDIterable,
            PrefetchConfig,
            CachedUPDZarrStoreBytes,
            ZarrByteCacheConfig,
        )
        return PhysicalSample, UPDZarrStore, PrefetchZarrUPDIterable, PrefetchConfig, CachedUPDZarrStoreBytes, ZarrByteCacheConfig
    except ModuleNotFoundError as e:
        # Most common: running from source without installing deps (e.g., zarr)
        missing = getattr(e, "name", None) or str(e)
        raise SystemExit(
            "\n".join(
                [
                    "[ERROR] Missing dependency while importing pinneaple_data.",
                    f"Missing module: {missing}",
                    "",
                    "Fix:",
                    "  1) Install project deps (recommended):",
                    "     pip install -e .",
                    "  2) Or install the missing package directly (common: zarr):",
                    "     pip install zarr",
                ]
            )
        ) from e


try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def sys_mem_mb() -> float:
    if psutil is None:
        return float("nan")
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def cuda_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def make_dataset(zarr_root: str, n: int, shape_t: int, shape_x: int) -> None:
    PhysicalSample, UPDZarrStore, *_ = _require_pinnea_data()

    if os.path.exists(zarr_root) and os.path.isdir(zarr_root):
        return
    os.makedirs(os.path.dirname(zarr_root), exist_ok=True)

    samples = []
    for i in range(n):
        u = torch.randn(shape_t, shape_x)
        samples.append(
            PhysicalSample(
                fields={"u": u},
                coords={},
                meta={"ids": {"sample_id": f"s{i:06d}"}},
            )
        )
    UPDZarrStore.write(zarr_root, samples, manifest={"bench": "data_io_bench", "count": n})
    print(f"[OK] Wrote Zarr dataset at {zarr_root} with {n} samples, u.shape=({shape_t},{shape_x})")


def _dtype(dtype: str) -> torch.dtype:
    return torch.float32 if dtype == "fp32" else torch.float16


def bench_plain_read(zarr_root: str, steps: int, device: str, dtype: str, latency_max: int) -> Dict[str, Any]:
    PhysicalSample, UPDZarrStore, *_ = _require_pinnea_data()

    store = UPDZarrStore(zarr_root, mode="r")
    n = store.count()
    dt = _dtype(dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    lat: List[float] = []
    t0 = time.perf_counter()
    checksum = 0.0
    count = 0

    for i in range(min(steps, n)):
        t1 = time.perf_counter()
        s = store.read_sample(i, fields=["u"], coords=[], device="cpu", dtype=dt)
        u = s.fields["u"]

        if device == "cuda":
            u = u.pin_memory().to("cuda", non_blocking=True)
        checksum += float(u.mean().item())

        t2 = time.perf_counter()
        if len(lat) < latency_max:
            lat.append((t2 - t1) * 1000.0)
        count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0
    return {
        "name": "plain_read",
        "samples": count,
        "seconds": dt_s,
        "samples_per_s": count / max(dt_s, 1e-9),
        "latency": latency_summary(lat),
        "sys_mem_mb": sys_mem_mb(),
        "cuda_peak_mb": cuda_mem_mb(),
        "checksum": checksum,
    }


def bench_cached_store_bytes(
    zarr_root: str, steps: int, device: str, dtype: str, max_mb: int, latency_max: int
) -> Dict[str, Any]:
    (
        PhysicalSample,
        UPDZarrStore,
        PrefetchZarrUPDIterable,
        PrefetchConfig,
        CachedUPDZarrStoreBytes,
        ZarrByteCacheConfig,
    ) = _require_pinnea_data()

    cache_cfg = ZarrByteCacheConfig(
        max_sample_bytes=max_mb * 1024 * 1024,
        max_field_bytes=max_mb * 1024 * 1024,
        enable_field_cache=True,
    )
    store = CachedUPDZarrStoreBytes(zarr_root, cache=cache_cfg, mode="r")

    n = store.count()
    dt = _dtype(dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    lat: List[float] = []
    t0 = time.perf_counter()
    checksum = 0.0
    count = 0

    # Two passes to show cache warmth.
    for _pass_id in range(2):
        for i in range(min(steps, n)):
            t1 = time.perf_counter()
            s = store.read_sample(i, fields=["u"], coords=[], device="cpu", dtype=dt)
            u = s.fields["u"]

            if device == "cuda":
                u = u.pin_memory().to("cuda", non_blocking=True)
            checksum += float(u.mean().item())

            t2 = time.perf_counter()
            if len(lat) < latency_max:
                lat.append((t2 - t1) * 1000.0)
            count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0
    stats = store.cache_stats()
    return {
        "name": "cached_bytes_store",
        "samples": count,
        "seconds": dt_s,
        "samples_per_s": count / max(dt_s, 1e-9),
        "latency": latency_summary(lat),
        "sys_mem_mb": sys_mem_mb(),
        "cuda_peak_mb": cuda_mem_mb(),
        "checksum": checksum,
        "cache_stats": stats,
    }


def bench_prefetch_iterable(
    zarr_root: str, steps: int, device: str, dtype: str, num_workers: int, pin: bool, latency_max: int
) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    (
        PhysicalSample,
        UPDZarrStore,
        PrefetchZarrUPDIterable,
        PrefetchConfig,
        CachedUPDZarrStoreBytes,
        ZarrByteCacheConfig,
    ) = _require_pinnea_data()

    dt = _dtype(dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefetch_cfg = PrefetchConfig(
        prefetch=16,
        queue_max=32,
        use_sample_cache=True,
        pin_memory=pin,
        target_device=("cuda" if device == "cuda" else "cpu"),
        transfer_non_blocking=True,
    )

    ds = PrefetchZarrUPDIterable(
        zarr_root,
        fields=["u"],
        coords=[],
        dtype=dt,
        cache=ZarrByteCacheConfig(
            max_sample_bytes=256 * 1024 * 1024,
            max_field_bytes=256 * 1024 * 1024,
            enable_field_cache=True,
        ),
        prefetch_cfg=prefetch_cfg,
    )
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers, persistent_workers=(num_workers > 0))

    lat: List[float] = []
    t0 = time.perf_counter()
    checksum = 0.0
    count = 0

    it = iter(dl)
    while count < steps:
        t1 = time.perf_counter()
        s = next(it)
        u = s.fields["u"]
        checksum += float(u.mean().item())
        t2 = time.perf_counter()
        if len(lat) < latency_max:
            lat.append((t2 - t1) * 1000.0)
        count += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt_s = time.perf_counter() - t0
    return {
        "name": "prefetch_iterable",
        "samples": count,
        "seconds": dt_s,
        "samples_per_s": count / max(dt_s, 1e-9),
        "latency": latency_summary(lat),
        "sys_mem_mb": sys_mem_mb(),
        "cuda_peak_mb": cuda_mem_mb(),
        "checksum": checksum,
        "config": asdict(prefetch_cfg),
        "num_workers": num_workers,
        "pin_memory": pin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bench_out/data_io")
    ap.add_argument("--zarr", type=str, default="bench_out/data_io/ds.zarr")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--X", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache_mb", type=int, default=512)
    ap.add_argument("--latency_max", type=int, default=2000, help="max samples to store for latency percentiles")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    make_dataset(args.zarr, args.n, args.T, args.X)

    results = []
    results.append(bench_plain_read(args.zarr, args.steps, args.device, args.dtype, args.latency_max))
    results.append(bench_cached_store_bytes(args.zarr, args.steps, args.device, args.dtype, args.cache_mb, args.latency_max))
    results.append(bench_prefetch_iterable(args.zarr, args.steps, args.device, args.dtype, args.workers, pin=True, latency_max=args.latency_max))
    results.append(bench_prefetch_iterable(args.zarr, args.steps, args.device, args.dtype, args.workers, pin=False, latency_max=args.latency_max))

    out_json = os.path.join(args.out, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "env": {
                    "torch": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "device": args.device,
                    "dtype": args.dtype,
                },
                "args": vars(args),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"[OK] Wrote {out_json}")
    for r in results:
        lat = r.get("latency", {})
        print(
            f" - {r['name']}: {r['samples_per_s']:.1f} samples/s | "
            f"p50={lat.get('p50_ms', float('nan')):.2f}ms p95={lat.get('p95_ms', float('nan')):.2f}ms"
        )


if __name__ == "__main__":
    main()