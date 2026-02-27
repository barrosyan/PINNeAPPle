from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional
import subprocess
import pandas as pd


def write_sample_dict_cloud(
    case_dir: str | Path,
    *,
    set_name: str,
    points: pd.DataFrame,          # columns: x,y,z
    fields: Sequence[str],         # ex: ["T"]
) -> Path:
    """
    Writes system/sampleDict for 'cloud' sampling at given points.
    """
    case_dir = Path(case_dir).resolve()
    sysdir = case_dir / "system"
    sysdir.mkdir(parents=True, exist_ok=True)

    pts = points[["x", "y", "z"]].to_numpy()

    # OpenFOAM expects: ( (x y z) (x y z) ... )
    pts_str = "\n".join([f"        ({float(x)} {float(y)} {float(z)})" for x, y, z in pts])

    fields_str = " ".join([str(f) for f in fields])

    txt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      sampleDict;
}}

setFormat raw;
surfaceFormat raw;

interpolationScheme cellPoint;

fields ({fields_str});

sets
(
    {set_name}
    {{
        type cloud;
        axis xyz;
        points
        (
{pts_str}
        );
    }}
);
"""
    out = sysdir / "sampleDict"
    out.write_text(txt)
    return out


def run_sampling(case_dir: str | Path, foam_bashrc: str, *, time: Optional[str] = None) -> None:
    """
    Runs: postProcess -func sample (latestTime by default)
    """
    case_dir = Path(case_dir).resolve()
    tflag = f"-time {time}" if time is not None else "-latestTime"
    cmd = f"bash -lc 'source {foam_bashrc} >/dev/null 2>&1 && cd \"{case_dir}\" && postProcess -func sample {tflag}'"
    subprocess.run(cmd, shell=True, check=True)


def _latest_sample_dir(case_dir: Path) -> Path:
    pp = case_dir / "postProcessing" / "sample"
    if not pp.exists():
        raise FileNotFoundError(f"No postProcessing/sample found under {case_dir}")
    times = sorted([p for p in pp.iterdir() if p.is_dir()], key=lambda p: float(p.name) if p.name.replace(".","",1).isdigit() else -1)
    if not times:
        raise FileNotFoundError(f"No time directories under {pp}")
    return times[-1]


def read_sampled_scalar_field(case_dir: str | Path, *, set_name: str, field: str) -> pd.DataFrame:
    """
    Reads postProcessing/sample/<time>/<set_name>_<field>.raw
    Format is typically: x y z value
    """
    case_dir = Path(case_dir).resolve()
    d = _latest_sample_dir(case_dir)
    f = d / f"{set_name}_{field}.raw"
    if not f.exists():
        raise FileNotFoundError(f"Sample output not found: {f}")

    df = pd.read_csv(f, delim_whitespace=True, header=None, names=["x","y","z",field])
    return df