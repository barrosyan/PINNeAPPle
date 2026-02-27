import argparse
from pathlib import Path
import json
import pandas as pd

from pinneaple_integrations.openfoam.case_builder import OpenFOAMCaseTemplate, stage_case_for_scenario
from pinneaple_integrations.openfoam.runner import OpenFOAMRunConfig, run_openfoam_case
from pinneaple_integrations.openfoam.sampling import (
    write_sample_dict_cloud,
    run_sampling,
    read_sampled_scalar_field,
)

def split_from_scenario_id(sid: int) -> str:
    # 14 train, 3 val, 3 test
    if sid < 14:
        return "train"
    if sid < 17:
        return "val"
    return "test"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--foam-env", required=True)
    ap.add_argument("--nprocs", type=int, default=1)
    args = ap.parse_args()

    proj = Path(args.proj).resolve()

    tpl = OpenFOAMCaseTemplate(
        template_dir=proj / "openfoam_template",
        obstacle_stl=proj / "geometry" / "obstacle_cylinder.stl",
    )

    conditions = json.loads((proj / "specs" / "conditions.json").read_text())
    scenarios = conditions["scenarios"]

    runs_dir = proj / "openfoam_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    sensors_points = pd.read_parquet(proj / "derived" / "sensors_points.parquet")
    line_points = pd.read_parquet(proj / "derived" / "line_points.parquet")

    all_cloud = []
    all_line = []

    cfg = OpenFOAMRunConfig(
        foam_bashrc=args.foam_env,
        solver="laplacianFoam",
        n_procs=int(args.nprocs),
        use_snappy=True,
    )

    for sc in scenarios:
        sid = int(sc["scenario_id"])
        case_dir = runs_dir / f"sc{sid:03d}"

        stage_case_for_scenario(tpl=tpl, out_case_dir=case_dir, scenario=sc)

        # sampleDict with TWO sets: we will overwrite sampleDict twice and sample twice
        # (simple and robust; if you want 2 sets in one file, you can extend helper)
        # 1) Cloud
        write_sample_dict_cloud(case_dir, set_name="pinneapleCloud", points=sensors_points, fields=["T"])
        run_openfoam_case(case_dir, cfg)
        run_sampling(case_dir, args.foam_env)

        dfT = read_sampled_scalar_field(case_dir, set_name="pinneapleCloud", field="T")
        dfT["scenario_id"] = sid
        dfT["split"] = split_from_scenario_id(sid)
        all_cloud.append(dfT)

        # 2) Line (same case, just resample)
        write_sample_dict_cloud(case_dir, set_name="pinneapleLine", points=line_points, fields=["T"])
        run_sampling(case_dir, args.foam_env)
        dfl = read_sampled_scalar_field(case_dir, set_name="pinneapleLine", field="T")
        dfl["scenario_id"] = sid
        dfl["split"] = split_from_scenario_id(sid)
        all_line.append(dfl)

        print(f"[OK] scenario {sid:03d}")

    datasets = proj / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)

    cloud = pd.concat(all_cloud, ignore_index=True)
    cloud.to_parquet(datasets / "sensors.parquet", index=False)

    line = pd.concat(all_line, ignore_index=True)
    line.to_parquet(datasets / "line_sensors.parquet", index=False)

    print("Wrote datasets/sensors.parquet and datasets/line_sensors.parquet")

if __name__ == "__main__":
    main()