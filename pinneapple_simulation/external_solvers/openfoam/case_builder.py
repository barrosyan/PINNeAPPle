from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import shutil


@dataclass(frozen=True)
class OpenFOAMCaseTemplate:
    template_dir: Path           # projects/.../openfoam_template
    obstacle_stl: Path           # projects/.../geometry/obstacle_cylinder.stl

    def validate(self) -> None:
        if not self.template_dir.exists():
            raise FileNotFoundError(self.template_dir)
        if not (self.template_dir / "system" / "blockMeshDict").exists():
            raise FileNotFoundError(self.template_dir / "system" / "blockMeshDict")
        if not (self.template_dir / "system" / "snappyHexMeshDict").exists():
            raise FileNotFoundError(self.template_dir / "system" / "snappyHexMeshDict")
        if not (self.template_dir / "0" / "T").exists():
            raise FileNotFoundError(self.template_dir / "0" / "T")
        if not self.obstacle_stl.exists():
            raise FileNotFoundError(self.obstacle_stl)


def stage_case_for_scenario(
    *,
    tpl: OpenFOAMCaseTemplate,
    out_case_dir: Path,
    scenario: Dict[str, Any],   # expects T_inlet, T_outlet, k, scenario_id
) -> None:
    """
    Copies template -> out_case_dir and writes scenario-specific 0/T and transportProperties.
    """
    tpl.validate()
    out_case_dir.mkdir(parents=True, exist_ok=True)
    # Copy template tree
    shutil.copytree(tpl.template_dir, out_case_dir, dirs_exist_ok=True)

    # Copy obstacle STL for snappy
    tri = out_case_dir / "constant" / "triSurface"
    tri.mkdir(parents=True, exist_ok=True)
    shutil.copy(tpl.obstacle_stl, tri / "obstacle.stl")

    Tin = float(scenario["T_inlet"])
    Tout = float(scenario.get("T_outlet", 0.0))
    # "k" is kept only as a legacy alias: this value is used directly as the
    # laplacianFoam diffusivity DT (m^2/s), NOT as thermal conductivity
    # (W/(m*K)) — the scenario schema carries no density/specific-heat to
    # convert conductivity to diffusivity. Prefer "diffusivity"/"alpha".
    diffusivity = float(
        scenario.get("diffusivity", scenario.get("alpha", scenario.get("k", 1.0)))
    )

    # Write 0/T with scenario values
    Ttxt = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {Tout};

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {Tin};
    }}
    outlet
    {{
        type fixedValue;
        value uniform {Tout};
    }}
    walls
    {{
        type zeroGradient;
    }}
    obstacle
    {{
        type zeroGradient;
    }}
}}
"""
    (out_case_dir / "0" / "T").write_text(Ttxt)

    # transportProperties: laplacianFoam uses diffusivity "DT" [m^2/s].
    tr = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
DT  [0 2 -1 0 0 0 0]  {diffusivity};
"""
    (out_case_dir / "constant" / "transportProperties").write_text(tr)