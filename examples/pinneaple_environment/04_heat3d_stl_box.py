from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

from pinneaple_environment import steady_heat_conduction_3d_default
from pinneaple_geom.builders.stl_domain_batch_builder import STLDomainBatchBuilder, STLDomainBatchConfig
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_models.pinns.vanilla import VanillaPINN


def make_box_stl(path: Path, *, extents=(1.0, 0.5, 0.25)) -> Path:
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def main():
    device = "cpu"

    # 1) Spec
    spec = steady_heat_conduction_3d_default()

    # Optional: override the volumetric source (Poisson RHS) via ctx['source_fn']
    # Here we create a small heater spot near the "inlet" plane (heuristic tag) for demo.
    def source_fn(X: np.ndarray, ctx):
        # X is (N,3) in normalized unit-box coordinates (builder normalizes by default)
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        hot = (x < -0.6) & (y * y + z * z < 0.08)
        return (5.0 * hot.astype(np.float32))[:, None]

    # 2) Loss
    loss_fn = compile_problem(spec, weights=LossWeights(w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=1.0))

    # 3) Geometry -> batch
    stl_path = make_box_stl(Path("/tmp/pinneaple_box.stl"))

    builder = STLDomainBatchBuilder(
        STLDomainBatchConfig(
            n_col=60_000,
            n_bc=24_000,
            inside_mode="trimesh_contains",
            device=device,
        )
    )

    batch = builder.build(spec, stl_path, user_ctx={"source_fn": source_fn})

    # 4) Model
    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(128, 128, 128, 128),
        activation="tanh",
    ).to(device)

    # 5) One forward loss
    out = loss_fn(model, None, batch)
    print("loss keys:", list(out.keys()))
    print("total:", float(out["total"].detach().cpu()))
    print("mesh_info:", batch["ctx"].get("mesh_info"))
    print("tags:", {k: int(v.sum()) for k, v in batch["ctx"].get("tag_masks", {}).items()})
    print("warnings (first 5):", batch["ctx"].get("warnings", [])[:5])


if __name__ == "__main__":
    main()
