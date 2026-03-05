from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pinneaple_environment.presets.industry import steady_heat_conduction_3d_default
from pinneaple_geom.builders.stl_domain_batch_builder import STLDomainBatchBuilder, STLDomainBatchConfig
from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_pinn.compiler import LossWeights, compile_problem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", type=str, required=True, default="examples/end_to_end/assets/Dragon 2.5_stl.stl", help="Path to STL file")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--n_col", type=int, default=80_000)
    ap.add_argument("--n_bc", type=int, default=30_000)
    ap.add_argument("--inside_mode", type=str, default="voxel_occupancy", choices=["trimesh_contains", "voxel_occupancy", "bbox", "sdf"])
    args = ap.parse_args()

    stl_path = Path(args.stl).expanduser().resolve()
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)

    device = args.device

    # 1) Define problem
    spec = steady_heat_conduction_3d_default()

    # 2) Compile loss from ProblemSpec
    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=1.0),
    )

    # 3) Build batch from STL
    builder = STLDomainBatchBuilder(
        STLDomainBatchConfig(
            n_col=args.n_col,
            n_bc=args.n_bc,
            inside_mode=args.inside_mode,
            device=device,
        )
    )
    batch = builder.build(spec, stl_path)

    # 4) Model
    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(128, 128, 128, 128),
        activation="tanh",
    ).to(device)

    # 5) One forward loss sanity-check
    out = loss_fn(model, None, batch)
    print("loss keys:", list(out.keys()))
    print("total:", float(out["total"].detach().cpu()))

    # 6) Minimal training loop example (optional)
    # You can plug this batch into pinneaple_train.Trainer by creating a Dataset/DataLoader.
    print("ctx warnings (first 5):", batch["ctx"].get("warnings", [])[:5])


if __name__ == "__main__":
    main()