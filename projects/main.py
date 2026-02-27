import argparse
from pathlib import Path
import subprocess

def run(cmd: str):
    print(f"\n[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="projects/heat_channel_obstacle_3d")
    ap.add_argument("--foam-env", required=True, help="OpenFOAM bashrc, ex: /opt/openfoam10/etc/bashrc")
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    proj = Path(args.project)

    run(f"python scripts/pipeline/01_make_geometry_channel_obstacle.py --out {proj}/geometry")
    run(f"python scripts/pipeline/02_make_points_from_stl.py --proj {proj}")
    run(f"python scripts/pipeline/03_make_specs_20_scenarios.py --proj {proj}")

    run(
        "python scripts/pipeline/04_openfoam_20_scenarios_sample.py "
        f"--proj {proj} --foam-env \"{args.foam_env}\" --nprocs {args.nprocs}"
    )

    run(f"python scripts/pipeline/05_export_bundle.py --proj {proj} --bundle {proj}/bundle_v1")

    run(f"python scripts/pipeline/06_train_pinn_models.py --proj {proj} {'--ddp' if args.ddp else ''}")
    run(f"python scripts/pipeline/07_train_operator_models.py --proj {proj}")

    run(f"python scripts/pipeline/08_evaluate_and_rank.py --proj {proj}")

if __name__ == "__main__":
    main()