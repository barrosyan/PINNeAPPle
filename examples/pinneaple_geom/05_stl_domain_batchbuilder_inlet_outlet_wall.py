"""From STL -> PINN-ready training batch (collocation + boundary) with auto-tags.

What this shows
--------------
- Generate a simple channel-like geometry (box) and export to STL.
- Use `STLDomainBatchBuilder` to sample:
  - interior collocation points (x_col)
  - boundary points + normals (x_bc, n_bc)
- Heuristic tagging of boundary points into inlet/outlet/wall using bbox planes.
- Attach boundary conditions via `ConditionSpec` that use those tags.

Run
---
python examples/pinneaple_geom/05_stl_domain_batchbuilder_inlet_outlet_wall.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinneaple_environment.conditions import DirichletBC
from pinneaple_environment.spec import PDETermSpec, ProblemSpec

from pinneaple_geom.gen.primitives import build_primitive
from pinneaple_geom.builders.stl_domain_batch_builder import STLDomainBatchBuilder, STLDomainBatchConfig


def _write_stl(mesh, path: Path) -> None:
    import trimesh

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(path)


def main() -> None:
    out_dir = Path("examples/pinneaple_geom/_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # A rectangular channel aligned with x-axis.
    # inlet: min-x plane, outlet: max-x plane, walls: everything else.
    mesh = build_primitive("box", extents=(4.0, 1.0, 1.0))
    stl_path = out_dir / "channel_box.stl"
    _write_stl(mesh, stl_path)

    # Simple BCs (toy):
    # - inlet: u=1, v=0, p=0
    # - outlet: p=0
    # In practice you'd likely use more realistic profiles.
    inlet = DirichletBC(
        name="inlet",
        fields=("u", "v", "p"),
        selector_type="tag",
        selector={"tag": "inlet"},
        value_fn=lambda X, ctx: np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (X.shape[0], 1)),
    )
    outlet = DirichletBC(
        name="outlet",
        fields=("p",),
        selector_type="tag",
        selector={"tag": "outlet"},
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
    )

    spec = ProblemSpec(
        name="toy_channel_3d",
        dim=3,
        coords=("x", "y", "z"),
        fields=("u", "v", "p"),
        pde=PDETermSpec(kind="navier_stokes_incompressible", fields=("u", "v", "p"), coords=("x", "y", "z")),
        conditions=(inlet, outlet),
    )

    cfg = STLDomainBatchConfig(
        n_col=40_000,
        n_bc=20_000,
        inside_mode="voxel_occupancy",  # robust even when not perfectly watertight
        device="cpu",
    )

    builder = STLDomainBatchBuilder(cfg)
    batch = builder.build(spec, stl_path)

    ctx = batch["ctx"]
    print("mesh_info:", ctx.get("mesh_info"))
    print("warnings:", ctx.get("warnings"))

    # Tag distribution on boundary samples
    tags = ctx.get("tag_masks", {})
    if tags:
        for k, m in tags.items():
            m = np.asarray(m)
            print(f"tag '{k}': {int(m.sum())} / {m.size}")

    # Tensor shapes
    print("x_col:", tuple(batch["x_col"].shape))
    print("x_bc:", tuple(batch["x_bc"].shape), "n_bc:", tuple(batch["n_bc"].shape))
    print("y_bc:", tuple(batch["y_bc"].shape))

    # Masks injected for each condition
    print("mask_inlet:", tuple(batch["mask_inlet"].shape), "true:", int(batch["mask_inlet"].sum().item()))
    print("mask_outlet:", tuple(batch["mask_outlet"].shape), "true:", int(batch["mask_outlet"].sum().item()))


if __name__ == "__main__":
    main()
