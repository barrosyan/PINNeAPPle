
"""Example: STEP/SDF geometry -> mesh -> point cloud + simple optimization loop.

This script shows three workflows:

1) STEP -> mesh (gmsh + meshio required)
2) Parametric SDF (2D) -> boundary point cloud (no gmsh needed)
3) Optimization loop: tune SDF params to match a target (toy objective)

Run:
  python examples/geom_step_pointcloud_opt_loop.py

Optional deps:
  pip install gmsh meshio scikit-image cma
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pinneaple_geom import (
    step_to_mesh,
    StepImportConfig,
    mesh_to_pointcloud,
    sdf2d_to_pointcloud,
    SDFGrid2D,
    ParamSpace,
    GeometryOptimizer,
)


def sdf_rounded_rect(params: dict[str, float]):
    # signed distance to a rounded rectangle centered at 0
    a = float(params.get("a", 0.6))  # half-width
    b = float(params.get("b", 0.4))  # half-height
    r = float(params.get("r", 0.12)) # corner radius

    def sdf(p: np.ndarray) -> np.ndarray:
        x = np.abs(p[:, 0]) - a
        y = np.abs(p[:, 1]) - b
        # outside distance
        ox = np.maximum(x, 0.0)
        oy = np.maximum(y, 0.0)
        outside = np.sqrt(ox * ox + oy * oy)
        inside = np.minimum(np.maximum(x, y), 0.0)
        return outside + inside - r

    return sdf


def demo_step():
    step_file = Path("./assets/example.step")
    if not step_file.exists():
        print("[STEP demo] No STEP file at ./assets/example.step (skipping).\n")
        return

    mesh = step_to_mesh(step_file, cfg=StepImportConfig(kind="surface", mesh_size=0.02), cache_dir="./_cache")
    pc = mesh_to_pointcloud(mesh, n_surface=4096, seed=0)
    print("[STEP demo] mesh vertices:", mesh.vertices.shape, "faces:", mesh.faces.shape)
    print("[STEP demo] point cloud:", tuple(pc.points.shape))


def demo_sdf_pointcloud():
    sdf = sdf_rounded_rect({"a": 0.6, "b": 0.35, "r": 0.10})
    pc = sdf2d_to_pointcloud(sdf, grid=SDFGrid2D(bounds_min=(-1,-1), bounds_max=(1,1), resolution=256), n_boundary=4096, band=0.01, seed=0)
    print("[SDF demo] boundary point cloud:", tuple(pc.points.shape))


def demo_opt_loop():
    # target: match a desired area proxy (toy)
    target_area = 0.6  # arbitrary

    space = ParamSpace(
        bounds={
            "a": (0.2, 0.9),
            "b": (0.2, 0.9),
            "r": (0.02, 0.25),
        },
        x0={"a": 0.55, "b": 0.35, "r": 0.10},
    )

    grid = SDFGrid2D(bounds_min=(-1,-1), bounds_max=(1,1), resolution=256)

    def area_proxy(params: dict[str, float]) -> float:
        sdf = sdf_rounded_rect(params)
        # approximate area by counting negative SDF on a grid
        xs = np.linspace(grid.bounds_min[0], grid.bounds_max[0], grid.resolution)
        ys = np.linspace(grid.bounds_min[1], grid.bounds_max[1], grid.resolution)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
        d = sdf(pts).reshape(grid.resolution, grid.resolution)
        inside = (d < 0).mean()
        # scale by domain area
        dom_area = (grid.bounds_max[0]-grid.bounds_min[0]) * (grid.bounds_max[1]-grid.bounds_min[1])
        area = inside * dom_area
        return float((area - target_area) ** 2)

    opt = GeometryOptimizer(space, seed=0, sigma0=0.15)

    def on_step(st):
        if st.step % 5 == 0:
            print(f"[OPT] step={st.step:03d} best_y={st.best_y:.6f} best_x={st.best_x}")

    st = opt.run(area_proxy, iters=25, batch=6, on_step=on_step)
    print("[OPT] final:", st.best_x, "loss:", st.best_y)


if __name__ == "__main__":
    demo_step()
    demo_sdf_pointcloud()
    demo_opt_loop()
