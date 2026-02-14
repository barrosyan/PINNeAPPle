from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = {
    "task_id": "flow_obstacle_2d",
    "version": "v0",
    "regions": ["inlet", "outlet", "walls", "obstacle"],
    "inlet": {"type": "dirichlet", "targets": {"u": 1.0, "v": 0.0}},
    "outlet": {"type": "dirichlet", "targets": {"p": 0.0}},
    "walls": {"type": "dirichlet", "targets": {"u": 0.0, "v": 0.0}},
    "obstacle": {"type": "dirichlet", "targets": {"u": 0.0, "v": 0.0}},
}

MANIFEST = {
    "bundle_schema_version": "1.0",
    "problem_id": "flow_obstacle_2d",
    "pde": {"name": "navier_stokes_2d", "time_dependent": False, "incompressible": True, "steady": True},
    "fields": {"inputs": ["x", "y"], "outputs": ["u", "v", "p"]},
    "nu": 0.01,
    "rho": 1.0,
    "domain": {"x": [0.0, 2.0], "y": [0.0, 1.0]},
    "geometry": {
        "type": "channel_with_circular_obstacle",
        "obstacle_circle": {"cx": 0.7, "cy": 0.5, "r": 0.15},
    },
    "regions": {
        "inlet": {"kind": "boundary", "description": "Channel inlet boundary"},
        "outlet": {"kind": "boundary", "description": "Channel outlet boundary"},
        "walls": {"kind": "boundary", "description": "Channel walls (no-slip)"},
        "obstacle": {"kind": "boundary", "description": "Obstacle boundary (no-slip)"},
        "interior": {"kind": "domain", "description": "Fluid interior excluding obstacle"},
    },
    "weights": {"pde": 1.0, "bc": 10.0, "data": 0.0},
    "sampling_plan": {"n_collocation_target": 200000, "n_boundary_target": 20000},
    "units": {"length": "m", "velocity": "m/s", "pressure": "Pa", "kinematic_viscosity": "m^2/s"},
    "notes": {
        "mvp": "constant inlet (u=1,v=0), outlet p=0, no-slip on walls and obstacle",
        "derived_points_required": ["derived/points_collocation.parquet", "derived/points_boundary.parquet"],
    },
}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-root", default="data/bundles/flow_obstacle_2d/v0", help="Bundle root directory")
    args = ap.parse_args()

    root = Path(args.bundle_root)
    out_dir = root / "bundle"
    write_json(out_dir / "conditions.json", CONDITIONS)
    write_json(out_dir / "manifest.json", MANIFEST)

    print(f"[OK] Wrote:\n- {out_dir / 'conditions.json'}\n- {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
