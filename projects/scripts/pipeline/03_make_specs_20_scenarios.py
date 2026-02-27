import argparse
from pathlib import Path
import json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    args = ap.parse_args()

    proj = Path(args.proj)
    specs = proj / "specs"
    specs.mkdir(parents=True, exist_ok=True)

    Tin_vals = np.linspace(0.80, 1.18, 20).tolist()
    scenarios = []
    for i, Tin in enumerate(Tin_vals):
        scenarios.append({"scenario_id": i, "T_inlet": float(Tin), "T_outlet": 0.0, "k": 1.0})

    manifest = {
        "problem_id": "heat3d_channel_obstacle_laplace",
        "pde": "laplace",
        "domain": {"dim": 3, "units": "SI_like"},
        "fields": ["T"],
        "weights": {"pde": 1.0, "bc": 50.0, "data": 1.0}
    }

    conditions = {
        "required_regions": ["inlet", "outlet", "walls", "obstacle"],
        "regions": {
            "inlet": {"kind": "dirichlet", "field": "T"},
            "outlet": {"kind": "dirichlet", "field": "T", "value": 0.0},
            "walls": {"kind": "neumann", "field": "T", "value": 0.0},
            "obstacle": {"kind": "neumann", "field": "T", "value": 0.0}
        },
        "scenarios": scenarios
    }

    (specs / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (specs / "conditions.json").write_text(json.dumps(conditions, indent=2))
    print("Wrote specs/manifest.json and specs/conditions.json")

if __name__ == "__main__":
    main()