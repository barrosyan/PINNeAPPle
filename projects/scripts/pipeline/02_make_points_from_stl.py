import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pinneaple_geom.builders.stl_domain_batch_builder import (
    STLDomainBatchBuilder,
    STLDomainBatchConfig,
    TagHeuristics,
)
from pinneaple_environment.spec import ProblemSpec
from pinneaple_environment.pdes import PDEspec


def region_from_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Regions in the physical domain:
      inlet:  x ~ 0
      outlet: x ~ 4
      obstacle: near cylinder surface around (y,z)=(0.5,0.5) with r~0.15 and x in [1.4,2.6]
      walls: remaining boundary
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    reg = np.full((xyz.shape[0],), "walls", dtype=object)

    inlet = x <= 1e-3
    outlet = x >= 4.0 - 1e-3

    # obstacle tube: inside x-range and close to radius
    r = np.sqrt((y - 0.5) ** 2 + (z - 0.5) ** 2)
    obstacle = (x >= 1.4 - 2e-2) & (x <= 2.6 + 2e-2) & (np.abs(r - 0.15) <= 2.5e-2)

    reg[inlet] = "inlet"
    reg[outlet] = "outlet"
    reg[obstacle] = "obstacle"
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--n_col", type=int, default=200_000)
    ap.add_argument("--n_bc", type=int, default=80_000)
    ap.add_argument("--n_sensors", type=int, default=30_000)
    ap.add_argument("--n_line", type=int, default=512)
    args = ap.parse_args()

    proj = Path(args.proj)
    stl = proj / "geometry" / "channel_minus_cylinder.stl"
    out = proj / "derived"
    out.mkdir(parents=True, exist_ok=True)

    # Minimal spec just to satisfy builder; we won't rely on its condition targets.
    spec = ProblemSpec(
        coords=("x", "y", "z"),
        fields=("T",),
        pde=PDEspec(kind="laplace", params={}),
        conditions=(),
    )

    cfg = STLDomainBatchConfig(
        n_col=int(args.n_col),
        n_bc=int(args.n_bc),
        normalize_unit_box=False,
        tags=TagHeuristics(enabled=False),  # we do our own region labeling
        device="cpu",
        dtype=torch.float32,
    )

    builder = STLDomainBatchBuilder(cfg)
    batch = builder.build(spec, stl_path=stl)

    x_col = batch["x_col"].cpu().numpy()
    x_bc = batch["x_bc"].cpu().numpy()
    n_bc = batch["n_bc"].cpu().numpy()

    df_col = pd.DataFrame(x_col, columns=["x", "y", "z"])
    df_col.to_parquet(out / "points_collocation.parquet", index=False)

    df_bc = pd.DataFrame(x_bc, columns=["x", "y", "z"])
    df_bc["nx"] = n_bc[:, 0]
    df_bc["ny"] = n_bc[:, 1]
    df_bc["nz"] = n_bc[:, 2]
    df_bc["region"] = region_from_xyz(x_bc)
    df_bc.to_parquet(out / "points_boundary.parquet", index=False)

    # sensors cloud: use random subset of collocation + boundary (more robust)
    rng = np.random.default_rng(7)
    col_idx = rng.choice(df_col.index.values, size=int(args.n_sensors), replace=False)
    sensors = df_col.loc[col_idx].copy()
    sensors.to_parquet(out / "sensors_points.parquet", index=False)

    # operator line points: centerline y=z=0.5, x uniform in [0,4]
    xs = np.linspace(0.0, 4.0, int(args.n_line), dtype=np.float32)
    line = pd.DataFrame({"x": xs, "y": np.full_like(xs, 0.5), "z": np.full_like(xs, 0.5)})
    line.to_parquet(out / "line_points.parquet", index=False)

    print("Generated:")
    print(" - derived/points_collocation.parquet")
    print(" - derived/points_boundary.parquet")
    print(" - derived/sensors_points.parquet")
    print(" - derived/line_points.parquet")


if __name__ == "__main__":
    main()