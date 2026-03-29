"""01 — Quickstart: run the built-in FlowObstacle2D task with the native backend.

What this demonstrates:
- Bundle loading + schema validation
- Backend training
- Task metric computation (PDE residuals + BC + optional sensors)
- Saving run artifacts + updating leaderboard

Run from repo root:
    python examples/pinneaple_arena/01_quickstart_native.py
"""

from __future__ import annotations

import json
import os
import time
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from pinneaple_arena.runner.run_benchmark import run_benchmark


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Atomic parquet write:
      - writes to a temp file
      - validates > 0 bytes
      - replaces target
    Prevents 0-byte parquet leftovers on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    try:
        # Write to temp
        df.to_parquet(tmp, index=False, engine="pyarrow")

        # Validate
        size = tmp.stat().st_size if tmp.exists() else 0
        if size <= 0:
            raise RuntimeError(f"Parquet temp file size is 0 bytes: {tmp}")

        # Replace target (atomic on Windows when possible)
        os.replace(tmp, path)

        # Final validation
        final_size = path.stat().st_size if path.exists() else 0
        if final_size <= 0:
            raise RuntimeError(f"Parquet final file size is 0 bytes: {path}")

    except Exception as e:
        # Cleanup temp if it exists
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise

def make_bundle_files(bundle_root: str | Path, *, seed: int = 123, n_collocation: int = 20000) -> Path:
    """
    Generates the required bundle v0 files for your loader/schema:

      bundle_root/
        bundle/
          manifest.json
          conditions.json
          sensors.parquet
          geometry.usd
        derived/
          points_collocation.parquet
          points_boundary.parquet
        geometry.usd  (duplicate at root for convenience)

    Domain: [0,1]x[0,1] with a circular obstacle.
    Regions: inlet, outlet, walls, obstacle (required by your schema defaults).
    """
    rng = np.random.default_rng(seed)
    root = Path(bundle_root)
    bundle_dir = root / "bundle"
    derived_dir = root / "derived"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    # --- Geometry params (match what you used earlier) ---
    cx, cy, r = 0.2, 0.5, 0.05

    # ---------------------------------------------------------
    # 1) derived/points_collocation.parquet  (requires x,y)
    # ---------------------------------------------------------
    # sample in [0,1]^2 excluding obstacle interior
    xs = rng.random(n_collocation * 3)
    ys = rng.random(n_collocation * 3)
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 >= r**2
    xs, ys = xs[mask][:n_collocation], ys[mask][:n_collocation]

    points_c = pd.DataFrame({"x": xs.astype("float32"), "y": ys.astype("float32")})
    _write_parquet(points_c, derived_dir / "points_collocation.parquet")

    # ---------------------------------------------------------
    # 2) derived/points_boundary.parquet  (requires x,y,region)
    # ---------------------------------------------------------
    n_edge = 4000
    n_ob = 4000

    y_in = rng.random(n_edge)
    y_out = rng.random(n_edge)
    x_w0 = rng.random(n_edge)
    x_w1 = rng.random(n_edge)

    inlet = pd.DataFrame(
        {"x": np.zeros(n_edge, dtype="float32"), "y": y_in.astype("float32"), "region": ["inlet"] * n_edge}
    )
    outlet = pd.DataFrame(
        {"x": np.ones(n_edge, dtype="float32"), "y": y_out.astype("float32"), "region": ["outlet"] * n_edge}
    )

    walls0 = pd.DataFrame({"x": x_w0.astype("float32"), "y": np.zeros(n_edge, dtype="float32"), "region": ["walls"] * n_edge})
    walls1 = pd.DataFrame({"x": x_w1.astype("float32"), "y": np.ones(n_edge, dtype="float32"), "region": ["walls"] * n_edge})

    theta = rng.random(n_ob) * 2 * np.pi
    x_ob = cx + r * np.cos(theta)
    y_ob = cy + r * np.sin(theta)
    obstacle = pd.DataFrame(
        {"x": x_ob.astype("float32"), "y": y_ob.astype("float32"), "region": ["obstacle"] * n_ob}
    )

    points_b = pd.concat([inlet, outlet, walls0, walls1, obstacle], ignore_index=True)
    _write_parquet(points_b, derived_dir / "points_boundary.parquet")

    # ---------------------------------------------------------
    # 3) bundle/sensors.parquet (optional unless require_sensors=true)
    # Required columns by schema defaults:
    #   x, y, u, v, scenario_id, split
    # ---------------------------------------------------------
    n_s = 1500
    xs = rng.random(n_s * 3)
    ys = rng.random(n_s * 3)
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 >= r**2
    xs, ys = xs[mask][:n_s], ys[mask][:n_s]

    # toy synthetic measurements (not physically perfect; just schema-correct)
    u = 4 * ys * (1 - ys) + 0.01 * rng.standard_normal(n_s)
    v = 0.01 * rng.standard_normal(n_s)

    split = np.array(["train"] * n_s, dtype=object)
    split[: int(0.1 * n_s)] = "val"
    split[int(0.1 * n_s) : int(0.2 * n_s)] = "test"

    sensors = pd.DataFrame(
        {
            "x": xs.astype("float32"),
            "y": ys.astype("float32"),
            "u": u.astype("float32"),
            "v": v.astype("float32"),
            "scenario_id": np.zeros(n_s, dtype="int32"),
            "split": split,
        }
    )
    _write_parquet(sensors, bundle_dir / "sensors.parquet")

    # ---------------------------------------------------------
    # 4) geometry.usd  (USD ASCII content, saved as .usd)
    # ---------------------------------------------------------
    usd_text = textwrap.dedent(
        f"""\
        #usda 1.0
        (
            defaultPrim = "World"
            metersPerUnit = 1
            upAxis = "Y"
        )

        def Xform "World"
        {{
            def Xform "Domain"
            {{
                # Rectangle domain [0,1]x[0,1] in XY plane
                def Mesh "DomainBoundary"
                {{
                    int[] faceVertexCounts = [4]
                    int[] faceVertexIndices = [0, 1, 2, 3]
                    point3f[] points = [
                        (0, 0, 0),
                        (1, 0, 0),
                        (1, 1, 0),
                        (0, 1, 0)
                    ]
                }}

                # Circular obstacle centered at ({cx}, {cy}), radius {r}
                def Xform "Obstacle"
                {{
                    def Sphere "ObstacleSphere"
                    {{
                        double radius = {r}
                        double3 xformOp:translate = ({cx}, {cy}, 0)
                        uniform token[] xformOpOrder = ["xformOp:translate"]
                    }}
                }}
            }}
        }}
        """
    ).strip() + "\n"

    # Save both at root and bundle/ (some schemas expect one or the other)
    (root / "geometry.usd").write_text(usd_text, encoding="utf-8")
    (bundle_dir / "geometry.usd").write_text(usd_text, encoding="utf-8")

    # ---------------------------------------------------------
    # Bonus: manifest.json + conditions.json (commonly required)
    # ---------------------------------------------------------
    manifest = {
        "problem_id": "flow_obstacle_2d",
        "pde": "incompressible_navier_stokes_2d",
        "nu": 0.001,
        "domain": {
            "type": "rectangle",
            "xlim": [0.0, 1.0],
            "ylim": [0.0, 1.0],
            "obstacle": {"type": "circle", "center": [cx, cy], "radius": r},
        },
        "fields": ["u", "v", "p"],
        "weights": {"pde": 1.0, "bc": 1.0, "div": 1.0},
    }
    conditions = {
        "inlet": {"type": "dirichlet", "u": 1.0, "v": 0.0},
        "outlet": {"type": "neumann", "p": 0.0},
        "walls": {"type": "no_slip", "u": 0.0, "v": 0.0},
        "obstacle": {"type": "no_slip", "u": 0.0, "v": 0.0},
        "_meta": {"generated_unix": time.time()},
    }

    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "conditions.json").write_text(json.dumps(conditions, indent=2), encoding="utf-8")

    return root


def main() -> None:

    repo_root = Path(__file__).resolve().parents[2]   

    artifacts_dir = repo_root / "data" / "artifacts" / "examples" / "quickstart_native"
    
    out = make_bundle_files(artifacts_dir)
    print(f"[OK] Generated bundle files at: {out.resolve()}")

    out = run_benchmark(
        artifacts_dir=artifacts_dir,
        task_cfg_path=repo_root / "examples" / "pinneaple_arena" / "configs" / "task_flow_obstacle_2d.yaml",
        run_cfg_path=repo_root / "examples" / "pinneaple_arena" / "configs" / "run_native_fast.yaml",
        bundle_schema_path=repo_root / "configs" / "data" / "bundle_schema.yaml",
    )

    print("\n=== DONE ===")
    print("run_id:", out["run_id"])
    print("run_dir:", out["run_dir"])
    print("summary:")
    for k, v in out["summary"]["key_metrics"].items():
        print(f"  - {k}: {v}")
    

if __name__ == "__main__":
    main()
