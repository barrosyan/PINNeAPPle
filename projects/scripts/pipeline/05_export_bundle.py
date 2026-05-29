import argparse
from pathlib import Path
from pinneapple_simulation.external_solvers.openfoam.export_bundle import export_bundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()

    export_bundle(project_dir=Path(args.proj), bundle_dir=Path(args.bundle))
    print(f"Exported bundle to {args.bundle}")

if __name__ == "__main__":
    main()