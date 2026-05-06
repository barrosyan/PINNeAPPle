"""
pinneaple_solvers: SolverCatalog + UPD adapters demo

What this shows
--------------
- Using SolverCatalog to build solvers by name (string registry).
- Using the UPD signal adapter to extract a 1D time signal from a
  multi-dimensional physical field in a *duck-typed* way.
- Running FFTSolver and retrieving frequency bins.

Run
---
python examples/pinneaple_solvers/05_solvers_catalog_and_adapters.py

Why duck-typed?
-------------
The adapter works with objects that *look like* a PhysicalSample.
This keeps the example lightweight (only torch dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any

import torch

from pinneaple_solvers.registry import SolverCatalog
from pinneaple_solvers.adapters.upd_signal import extract_1d_signal


@dataclass
class MiniSample:
    """Minimal PhysicalSample-like object for the adapter."""
    fields: Dict[str, torch.Tensor]
    axes: Dict[str, list[str]]
    meta: Dict[str, Any]


def main():
    torch.manual_seed(0)

    # Create a synthetic field: u(x,y,t) with a known dominant frequency in time
    H, W, T = 16, 16, 1024
    dt = 1.0 / 100.0
    t = torch.arange(T) * dt

    f_dom = 7.0  # Hz
    u_t = torch.sin(2 * math.pi * f_dom * t)
    u_xy = torch.randn(H, W) * 0.1
    u = u_xy[:, :, None] + u_t[None, None, :]  # (H,W,T)

    sample = MiniSample(
        fields={"u": u},
        axes={"u": ["y", "x", "t"]},
        meta={"dt": dt},
    )

    # 1) Adapter: extract a 1D time signal by reducing spatial dims
    sig, meta = extract_1d_signal(sample, var="u", axis="t", reduce="mean")
    print("--- adapter")
    print(meta)

    # 2) Build a solver via registry
    catalog = SolverCatalog()
    fft = catalog.build("fft", real_input=True, output="magnitude")

    out = fft(sig, d=dt)
    mag = out.result
    freqs = out.extras.get("freqs")

    # 3) Find dominant frequency peak (skip DC)
    k0 = 1
    k_peak = int(torch.argmax(mag[k0:]).item()) + k0
    f_peak = float(freqs[k_peak].item()) if freqs is not None else float("nan")

    print("--- fft")
    print(f"mag shape: {tuple(mag.shape)}")
    print(f"peak bin: k={k_peak} | f_peak={f_peak:.3f} Hz (expected ~{f_dom} Hz)")


if __name__ == "__main__":
    main()