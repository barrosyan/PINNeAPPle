"""04_synth_pde_end_to_end.py

Showcase: generate synthetic PDE/ODE trajectories -> convert to PhysicalSample ->
store as UPD Zarr -> stream back for training.

Demonstrates:
  - Synthetic generator: pinneaple_data.synth.pde.PDESynthGenerator
  - Conversion: pinneaple_data.synth.sample_adapter.to_physical_sample
  - Storage: pinneaple_data.zarr_store.UPDZarrStore
  - Streaming: pinneaple_data.zarr_iterable.ZarrUPDIterable

Run:
  python examples/pinneaple_data/04_synth_pde_end_to_end.py
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.synth.pde import PDESynthGenerator
from pinneaple_data.synth.sample_adapter import to_physical_sample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_iterable import ZarrUPDIterable


def _as_ps(sample_like: object, *, kind: str) -> PhysicalSample:
    ps = to_physical_sample(sample_like)

    # The synth adapter may return a PhysicalSample OR a minimal fallback.
    # Normalize into the main PhysicalSample schema.
    if isinstance(ps, PhysicalSample):
        ps.domain = {"type": "grid"}
        ps.schema = {"governing": {"type": "PDE/ODE", "name": kind}}
        ps.provenance = {"source": "pinneaple_data.synth", "kind": kind}
        return ps

    # Fallback: treat it as duck-typed with fields/coords/meta.
    state = dict(ps.fields)  # type: ignore[attr-defined]
    state["coords"] = dict(ps.coords)  # type: ignore[attr-defined]
    return PhysicalSample(
        state=state,
        domain={"type": "grid"},
        schema={"governing": {"type": "PDE/ODE", "name": kind}},
        provenance={"source": "pinneaple_data.synth", "kind": kind},
        extras={"meta": dict(ps.meta)},  # type: ignore[attr-defined]
    )


def main() -> None:
    out_dir = "examples/_out"
    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, "synth_pde_ds.zarr")

    if not os.path.isdir(zarr_path):
        gen = PDESynthGenerator(seed=7)
        samples = []

        specs = [
            ("heat1d", dict(T=64, X=64, alpha=0.05)),
            ("advection1d", dict(T=64, X=64, c=0.8)),
            ("logistic", dict(T=128, r=1.5, K=2.0)),
        ]

        for kind, cfg in specs:
            for _ in range(40):
                out = gen.sample(kind=kind, **cfg)
                samples.append(_as_ps(out.sample, kind=kind))

        UPDZarrStore.write(
            zarr_path,
            samples,
            manifest={
                "name": "synth_pde",
                "n": len(samples),
                "kinds": [k for k, _ in specs],
                "note": "heat1d/advection1d/logistic mixed",
            },
        )
        print(f"Wrote {len(samples)} samples -> {zarr_path}")

    ds = ZarrUPDIterable(zarr_path, fields=None, coords=None)
    dl = DataLoader(ds, batch_size=None, num_workers=2, persistent_workers=True)

    for i, s in enumerate(dl):
        keys = list(s.state.keys()) if isinstance(s.state, dict) else s.list_variables()
        print(f"[{i}] kind={s.provenance.get('kind')} vars={keys[:6]}...")

        # toy training prep: (t,x)->u if present
        if isinstance(s.state, dict) and "u" in s.state and "coords" in s.state:
            u = s.state["u"]
            coords = s.state["coords"]
            if isinstance(u, torch.Tensor) and isinstance(coords, dict) and "t" in coords:
                print("    u:", tuple(u.shape), "t:", tuple(coords["t"].shape))
        if i >= 5:
            break


if __name__ == "__main__":
    main()
