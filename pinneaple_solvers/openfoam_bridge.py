"""OpenFOAM bridge for pinneaple.

Generates OpenFOAM case directories from ProblemSpec, runs the solver,
and extracts field data as numpy arrays for dataset generation and
surrogate training.

Key functions
-------------
- generate_case(problem_spec, case_dir, mesh_cfg)  -> case directory
- run_openfoam(case_dir, solver, n_cores, n_iter)  -> CompletedProcess
- extract_fields(case_dir, time, field_names)       -> dict[str, np.ndarray]
- openfoam_to_dataset(case_dir, field_names, ...)   -> dict[str, np.ndarray]
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _which_openfoam(openfoam_bin: Optional[str] = None) -> Optional[Path]:
    """Return path to an OpenFOAM executable (blockMesh used as probe)."""
    probe = openfoam_bin or "blockMesh"
    found = shutil.which(probe)
    return Path(found) if found else None


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (documented in module docstring)
# ---------------------------------------------------------------------------

def generate_case(problem_spec: Any, case_dir: Path, mesh_cfg: Optional[Dict] = None) -> Path:
    """Generate an OpenFOAM case directory from *problem_spec*.

    Parameters
    ----------
    problem_spec:
        Object with at least `conditions` (dict) and optional `nu` (float).
    case_dir:
        Target directory (will be created if absent).
    mesh_cfg:
        Optional dict passed through to the bridge's generate_case method.

    Returns
    -------
    Path
        Resolved *case_dir*.
    """
    bridge = OpenFOAMBridge()
    return bridge.generate_case(problem_spec, Path(case_dir), **(mesh_cfg or {}))


def run_openfoam(
    case_dir: Path,
    solver: str = "simpleFoam",
    n_cores: int = 1,
    n_iter: int = 500,
) -> "subprocess.CompletedProcess[str]":
    """Run an OpenFOAM solver inside *case_dir*.

    Returns
    -------
    subprocess.CompletedProcess
        Result of the solver subprocess.
    """
    bridge = OpenFOAMBridge(solver=solver, n_cores=n_cores, n_iterations=n_iter)
    info = bridge.run_solver(Path(case_dir))
    # Return a lightweight stand-in so callers that inspect .returncode work.
    rc = 0 if info.get("converged", False) else 1
    return subprocess.CompletedProcess(args=solver, returncode=rc, stdout="", stderr="")


def extract_fields(
    case_dir: Path,
    time: str = "latestTime",
    field_names: Optional[List[str]] = None,
) -> Dict[str, "Any"]:
    """Extract field arrays from a finished OpenFOAM run.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from field name to numpy array (or empty dict on failure).
    """
    bridge = OpenFOAMBridge()
    return bridge.extract_fields(Path(case_dir), time=time, field_names=field_names)


def openfoam_to_dataset(
    case_dir: Path,
    field_names: Optional[List[str]] = None,
    time: str = "latestTime",
) -> Dict[str, "Any"]:
    """High-level helper: extract fields and return as a flat dataset dict."""
    return extract_fields(case_dir, time=time, field_names=field_names)


# ---------------------------------------------------------------------------
# OpenFOAM file templates
# ---------------------------------------------------------------------------

_CONTROL_DICT_TMPL = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

application     {application};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;

writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

_FV_SCHEMES_TMPL = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default     steadyState;
}

gradSchemes
{
    default     Gauss linear;
    grad(p)     Gauss linear;
    grad(U)     Gauss linear;
}

divSchemes
{
    default     none;
    div(phi,U)  bounded Gauss linearUpwind grad(U);
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default     Gauss linear corrected;
}

interpolationSchemes
{
    default     linear;
}

snGradSchemes
{
    default     corrected;
}
"""

_FV_SOLUTION_TMPL = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
}

relaxationFactors
{
    equations
    {
        U               0.9;
        p               0.7;
    }
}
"""

_TRANSPORT_PROPS_TMPL = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}

transportModel  Newtonian;
nu              nu [ 0 2 -1 0 0 0 0 ] {nu:.6g};
"""

_U_FIELD_TMPL = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
{bc_entries}}}
"""

_P_FIELD_TMPL = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{bc_entries}}}
"""


def _build_u_bc_entries(conditions: Dict[str, Any]) -> str:
    """Convert problem_spec.conditions to OpenFOAM U boundary entries."""
    lines: List[str] = []
    seen: set = set()
    for cond_name, cond in conditions.items():
        patch = getattr(cond, "patch", cond_name)
        if patch in seen:
            continue
        seen.add(patch)
        cond_type = getattr(cond, "type", "wall")
        if cond_type in ("inlet", "velocity_inlet"):
            vel = getattr(cond, "value", [1.0, 0.0, 0.0])
            if not isinstance(vel, (list, tuple)):
                vel = [float(vel), 0.0, 0.0]
            lines.append(
                f"    {patch}\n    {{\n"
                f"        type            fixedValue;\n"
                f"        value           uniform ({vel[0]} {vel[1]} {vel[2]});\n"
                f"    }}\n"
            )
        elif cond_type in ("outlet", "pressure_outlet"):
            lines.append(
                f"    {patch}\n    {{\n"
                f"        type            zeroGradient;\n"
                f"    }}\n"
            )
        else:  # wall / default
            lines.append(
                f"    {patch}\n    {{\n"
                f"        type            noSlip;\n"
                f"    }}\n"
            )
    if not lines:
        # Minimal fallback
        lines = [
            "    inlet\n    {\n        type            fixedValue;\n        value           uniform (1 0 0);\n    }\n",
            "    outlet\n    {\n        type            zeroGradient;\n    }\n",
            "    walls\n    {\n        type            noSlip;\n    }\n",
        ]
    return "".join(lines)


def _build_p_bc_entries(conditions: Dict[str, Any]) -> str:
    """Convert problem_spec.conditions to OpenFOAM p boundary entries."""
    lines: List[str] = []
    seen: set = set()
    for cond_name, cond in conditions.items():
        patch = getattr(cond, "patch", cond_name)
        if patch in seen:
            continue
        seen.add(patch)
        cond_type = getattr(cond, "type", "wall")
        if cond_type in ("outlet", "pressure_outlet"):
            p_val = getattr(cond, "pressure", 0.0)
            lines.append(
                f"    {patch}\n    {{\n"
                f"        type            fixedValue;\n"
                f"        value           uniform {float(p_val):.6g};\n"
                f"    }}\n"
            )
        else:
            lines.append(
                f"    {patch}\n    {{\n"
                f"        type            zeroGradient;\n"
                f"    }}\n"
            )
    if not lines:
        lines = [
            "    inlet\n    {\n        type            zeroGradient;\n    }\n",
            "    outlet\n    {\n        type            fixedValue;\n        value           uniform 0;\n    }\n",
            "    walls\n    {\n        type            zeroGradient;\n    }\n",
        ]
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@SolverRegistry.register(
    name="openfoam",
    family="pde",
    description="OpenFOAM CFD bridge",
    tags=["openfoam", "cfd", "fluid"],
)
class OpenFOAMBridge(SolverBase):
    """Bridge that generates OpenFOAM cases, runs the solver, and extracts fields.

    Parameters
    ----------
    solver:
        OpenFOAM application name (e.g. ``"simpleFoam"``, ``"icoFoam"``).
    n_cores:
        Number of MPI processes.  ``1`` runs serial (no decomposition).
    n_iterations:
        ``endTime`` written into ``system/controlDict``.
    openfoam_bin:
        Optional explicit path to the OpenFOAM binary directory.  When
        ``None`` the bridge probes ``PATH`` via ``shutil.which``.
    """

    def __init__(
        self,
        solver: str = "simpleFoam",
        n_cores: int = 1,
        n_iterations: int = 500,
        openfoam_bin: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.solver = solver
        self.n_cores = int(n_cores)
        self.n_iterations = int(n_iterations)
        self.openfoam_bin = openfoam_bin

    # ------------------------------------------------------------------
    # SolverBase interface
    # ------------------------------------------------------------------

    def forward(self, problem_spec: Any, case_dir: Any, **kwargs: Any) -> SolverOutput:  # type: ignore[override]
        """Generate case, run solver, extract fields, return SolverOutput.

        Parameters
        ----------
        problem_spec:
            Problem specification object.
        case_dir:
            Path (or str) for the OpenFOAM case directory.
        **kwargs:
            Extra keyword arguments forwarded to ``generate_case``.

        Returns
        -------
        SolverOutput
            ``result`` is a 1-D tensor of extracted field values concatenated
            in the order returned by ``extract_fields``.  ``extras`` contains
            convergence info and the raw field dict.
        """
        if _which_openfoam(self.openfoam_bin) is None:
            warnings.warn(
                "OpenFOAM not found on PATH. Returning empty SolverOutput. "
                "Install OpenFOAM and ensure its binaries are on PATH.",
                RuntimeWarning,
                stacklevel=2,
            )
            return SolverOutput(
                result=torch.empty(0),
                losses={},
                extras={"openfoam_available": False},
            )

        case_dir = Path(case_dir)
        self.generate_case(problem_spec, case_dir, **kwargs)
        conv_info = self.run_solver(case_dir)
        field_names = kwargs.get("field_names", None)
        fields = self.extract_fields(case_dir, field_names=field_names)

        # Flatten field arrays into a single tensor for the result slot.
        tensors = []
        try:
            import numpy as np
            for arr in fields.values():
                tensors.append(torch.as_tensor(arr.ravel(), dtype=torch.float32))
        except Exception:
            pass

        result = torch.cat(tensors) if tensors else torch.empty(0)

        return SolverOutput(
            result=result,
            losses={},
            extras={
                "openfoam_available": True,
                "convergence": conv_info,
                "fields": fields,
                "case_dir": str(case_dir),
            },
        )

    # ------------------------------------------------------------------
    # Case generation
    # ------------------------------------------------------------------

    def generate_case(self, problem_spec: Any, case_dir: Path, **kwargs: Any) -> Path:
        """Create a minimal OpenFOAM case directory from *problem_spec*.

        Creates the standard ``0/``, ``constant/``, ``system/`` layout with
        template files populated from *problem_spec* attributes.

        Parameters
        ----------
        problem_spec:
            Must expose ``conditions`` (dict-like) and optionally ``nu``
            (kinematic viscosity, default 1e-5).
        case_dir:
            Target root directory.

        Returns
        -------
        Path
            Resolved *case_dir*.
        """
        case_dir = Path(case_dir).resolve()
        case_dir.mkdir(parents=True, exist_ok=True)

        nu = float(getattr(problem_spec, "nu", 1e-5))
        conditions: Dict[str, Any] = getattr(problem_spec, "conditions", {}) or {}

        # -- system/controlDict --
        _write_file(
            case_dir / "system" / "controlDict",
            _CONTROL_DICT_TMPL.format(
                application=self.solver,
                end_time=self.n_iterations,
                write_interval=max(1, self.n_iterations // 10),
            ),
        )

        # -- system/fvSchemes --
        _write_file(case_dir / "system" / "fvSchemes", _FV_SCHEMES_TMPL)

        # -- system/fvSolution --
        _write_file(case_dir / "system" / "fvSolution", _FV_SOLUTION_TMPL)

        # -- constant/transportProperties --
        _write_file(
            case_dir / "constant" / "transportProperties",
            _TRANSPORT_PROPS_TMPL.format(nu=nu),
        )

        # -- 0/U --
        u_bc = _build_u_bc_entries(conditions)
        _write_file(case_dir / "0" / "U", _U_FIELD_TMPL.format(bc_entries=u_bc))

        # -- 0/p --
        p_bc = _build_p_bc_entries(conditions)
        _write_file(case_dir / "0" / "p", _P_FIELD_TMPL.format(bc_entries=p_bc))

        log.info("OpenFOAM case generated at %s", case_dir)
        return case_dir

    # ------------------------------------------------------------------
    # Running the solver
    # ------------------------------------------------------------------

    def run_solver(self, case_dir: Path) -> Dict[str, Any]:
        """Run ``blockMesh`` followed by the configured OpenFOAM solver.

        Parameters
        ----------
        case_dir:
            Root of a previously generated OpenFOAM case.

        Returns
        -------
        dict
            Keys: ``returncode``, ``converged`` (bool), ``stdout``, ``stderr``.
        """
        case_dir = Path(case_dir).resolve()
        info: Dict[str, Any] = {"returncode": -1, "converged": False, "stdout": "", "stderr": ""}

        def _run(cmd: List[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                cmd,
                cwd=str(case_dir),
                capture_output=True,
                text=True,
            )

        # blockMesh
        bm = _run(["blockMesh"])
        if bm.returncode != 0:
            log.error("blockMesh failed (rc=%d): %s", bm.returncode, bm.stderr)
            info["returncode"] = bm.returncode
            info["stderr"] = bm.stderr
            return info

        # Parallel decomposition when n_cores > 1
        if self.n_cores > 1:
            dec = _run(["decomposePar"])
            if dec.returncode != 0:
                log.warning("decomposePar failed; falling back to serial run.")

        # Main solver (serial or parallel)
        if self.n_cores > 1:
            solver_cmd = ["mpirun", "-np", str(self.n_cores), self.solver, "-parallel"]
        else:
            solver_cmd = [self.solver]

        proc = _run(solver_cmd)
        info["returncode"] = proc.returncode
        info["stdout"] = proc.stdout
        info["stderr"] = proc.stderr
        # Heuristic: check for "SIMPLE solution converged" in stdout
        info["converged"] = "solution converged" in proc.stdout.lower() or proc.returncode == 0
        if proc.returncode != 0:
            log.error("OpenFOAM solver '%s' failed (rc=%d).", self.solver, proc.returncode)
        else:
            log.info("OpenFOAM solver '%s' finished successfully.", self.solver)
        return info

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def extract_fields(
        self,
        case_dir: Path,
        time: str = "latestTime",
        field_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract field data from a finished OpenFOAM run.

        Attempts (in order):
        1. ``foamToVTK`` to convert fields then parse VTK files with VTK / pyvista.
        2. Naive ASCII parsing of OpenFOAM field files as a last resort.

        Parameters
        ----------
        case_dir:
            Root of the case directory.
        time:
            Time directory to read from.  ``"latestTime"`` selects the
            highest-numbered subdirectory.
        field_names:
            List of field names to extract (e.g. ``["U", "p"]``).
            ``None`` defaults to ``["U", "p"]``.

        Returns
        -------
        dict[str, np.ndarray]
            Field name -> numpy array.  Empty dict on failure.
        """
        try:
            import numpy as np
        except ImportError:
            warnings.warn("numpy is required for field extraction.", RuntimeWarning, stacklevel=2)
            return {}

        case_dir = Path(case_dir).resolve()
        field_names = field_names or ["U", "p"]

        # Resolve time directory
        time_dir = self._resolve_time_dir(case_dir, time)
        if time_dir is None:
            log.warning("No time directory found in %s", case_dir)
            return {}

        # Try foamToVTK path first
        vtk_fields = self._extract_via_vtk(case_dir, time_dir, field_names, np)
        if vtk_fields:
            return vtk_fields

        # Fallback: naive ASCII parsing
        return self._extract_ascii(time_dir, field_names, np)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_time_dir(self, case_dir: Path, time: str) -> Optional[Path]:
        """Return the Path of the requested time directory."""
        if time == "latestTime":
            candidates = []
            for p in case_dir.iterdir():
                if p.is_dir():
                    try:
                        candidates.append((float(p.name), p))
                    except ValueError:
                        pass
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
        td = case_dir / time
        return td if td.is_dir() else None

    def _extract_via_vtk(
        self,
        case_dir: Path,
        time_dir: Path,
        field_names: List[str],
        np: Any,
    ) -> Dict[str, Any]:
        """Run foamToVTK and parse result with pyvista or vtk."""
        vtk_dir = case_dir / "VTK"
        try:
            proc = subprocess.run(
                ["foamToVTK", "-time", time_dir.name],
                cwd=str(case_dir),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not vtk_dir.exists():
                return {}
        except FileNotFoundError:
            return {}

        try:
            import pyvista as pv  # type: ignore[import]
        except ImportError:
            return {}

        fields: Dict[str, Any] = {}
        for vtk_file in sorted(vtk_dir.rglob("*.vtk")):
            try:
                mesh = pv.read(str(vtk_file))
                for fname in field_names:
                    if fname in mesh.array_names and fname not in fields:
                        fields[fname] = np.array(mesh[fname])
            except Exception as exc:
                log.debug("pyvista read failed for %s: %s", vtk_file, exc)
        return fields

    def _extract_ascii(
        self,
        time_dir: Path,
        field_names: List[str],
        np: Any,
    ) -> Dict[str, Any]:
        """Naive parser for OpenFOAM ASCII field files (internalField only)."""
        fields: Dict[str, Any] = {}
        for fname in field_names:
            fpath = time_dir / fname
            if not fpath.exists():
                continue
            try:
                text = fpath.read_text(errors="replace")
                # Find internalField block
                start = text.find("internalField")
                if start == -1:
                    continue
                block = text[start:]
                # Uniform scalar: internalField uniform 0;
                if "uniform" in block.split("\n")[0]:
                    parts = block.split()
                    try:
                        val_str = parts[2].rstrip(";").strip("()")
                        fields[fname] = np.array([float(v) for v in val_str.split()])
                    except (ValueError, IndexError):
                        pass
                    continue
                # Non-uniform list
                paren = block.find("(")
                end_paren = block.find(")", paren)
                if paren == -1:
                    continue
                raw = block[paren + 1:end_paren].strip()
                values: List[float] = []
                for token in raw.split():
                    try:
                        values.append(float(token))
                    except ValueError:
                        pass
                if values:
                    fields[fname] = np.array(values)
            except Exception as exc:
                log.debug("ASCII parse failed for %s: %s", fpath, exc)
        return fields
