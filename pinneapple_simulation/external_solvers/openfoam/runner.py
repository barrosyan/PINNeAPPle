from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import subprocess


@dataclass(frozen=True)
class OpenFOAMRunConfig:
    foam_bashrc: str                 # ex: "/opt/openfoam10/etc/bashrc"
    solver: str = "laplacianFoam"    # heat steady
    n_procs: int = 1                 # mpirun if >1
    use_snappy: bool = True
    extra_cmds: Sequence[str] = ()   # hooks


def _bash(foam_bashrc: str, cmd: str, cwd: Path) -> None:
    full = f"bash -lc 'source {foam_bashrc} >/dev/null 2>&1 && cd \"{cwd}\" && {cmd}'"
    subprocess.run(full, shell=True, check=True)


def run_openfoam_case(case_dir: str | Path, cfg: OpenFOAMRunConfig) -> None:
    case_dir = Path(case_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    # Mesh
    _bash(cfg.foam_bashrc, "blockMesh", case_dir)
    if cfg.use_snappy:
        _bash(cfg.foam_bashrc, "snappyHexMesh -overwrite", case_dir)

    # Solve
    if cfg.n_procs > 1:
        _bash(cfg.foam_bashrc, f"decomposePar -force", case_dir)
        _bash(cfg.foam_bashrc, f"mpirun -np {cfg.n_procs} {cfg.solver} -parallel", case_dir)
        _bash(cfg.foam_bashrc, "reconstructPar -latestTime", case_dir)
    else:
        _bash(cfg.foam_bashrc, cfg.solver, case_dir)

    # Extra hooks (optional)
    for c in cfg.extra_cmds:
        _bash(cfg.foam_bashrc, c, case_dir)