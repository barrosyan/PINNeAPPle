from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

from pinneaple_environment import linear_elasticity_3d_default
from pinneaple_geom.builders.stl_domain_batch_builder import STLDomainBatchBuilder, STLDomainBatchConfig, TagHeuristics
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_models.pinns.vanilla import VanillaPINN


def make_beam_stl(path: Path, *, extents=(2.0, 0.4, 0.4)) -> Path:
    """Simple cantilever-like beam as a box mesh exported to STL."""
    import trimesh

    mesh = trimesh.creation.box(extents=extents)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def main():
    device = "cpu"

    # 1) Spec: linear elasticity with a fixed support tag
    spec = linear_elasticity_3d_default()

    # 2) Provide a constant body force (e.g., gravity in -y)
    def body_force_fn(X: np.ndarray, ctx):
        b = np.zeros((X.shape[0], 3), dtype=np.float32)
        b[:, 1] = -0.5
        return b

    loss_fn = compile_problem(spec, weights=LossWeights(w_pde=1.0, w_bc=5.0, w_ic=1.0, w_data=1.0))

    # 3) Geometry batch from STL
    stl_path = make_beam_stl(Path("/tmp/pinneaple_beam.stl"))

    cfg = STLDomainBatchConfig(
        # NOTE: Elasticity loss can be expensive (multiple gradients).
        # Start tiny for a quick sanity-check, then scale up + consider GPU.
        n_col=64,
        n_bc=64,
        inside_mode="trimesh_contains",
        device=device,
    )

    # Tag heuristics: mark the *x-min* plane as "fixed" (matches the preset)
    cfg.tags = TagHeuristics(
        enabled=True,
        inlet_outlet_axis=0,
        inlet_is_min=True,
        outlet_is_min=False,
        fixed_axis=0,
        fixed_is_min=True,
    )

    builder = STLDomainBatchBuilder(cfg)
    batch = builder.build(spec, stl_path, user_ctx={"body_force_fn": body_force_fn})
    print("built batch", batch["x_col"].shape, batch["x_bc"].shape)

    # 4) Model (ux,uy,uz)
    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(16, 16),
        activation="tanh",
    ).to(device)

    print("starting loss eval")
    out = loss_fn(model, None, batch)
    print("loss keys:", list(out.keys()))
    print("total:", float(out["total"].detach().cpu()))
    print("tags:", {k: int(v.sum()) for k, v in batch["ctx"].get("tag_masks", {}).items()})


if __name__ == "__main__":
    main()
