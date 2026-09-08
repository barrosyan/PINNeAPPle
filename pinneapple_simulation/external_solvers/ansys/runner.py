"""ANSYS Fluent live-solver coupling: launches a real ``fluent`` batch-mode
process against a generated journal file and drives an actual solve, then
hands the result off to the existing ``cfd_formats.fluent_mesh_reader``
UPD bridge.

Mirrors ``external_solvers/openfoam/runner.py``'s shape (a frozen
``RunConfig``-style dataclass, subprocess invocation of a real external
executable, explicit failure if the executable isn't found) rather than
inventing a different structure for this second live-solver integration.

Fluent's batch-mode command line (``fluent <precision> -g -i
<journal>.jou [-t<nprocs>]``) is Fluent's own public, documented launch
syntax -- ``-g`` (no GUI), ``-i <journal>`` (run a journal file
non-interactively), ``-t<n>`` (processor count) are all in ANSYS's own
Fluent User's Guide / Text Command List documentation and used throughout
publicly published Fluent batch-run tutorials and HPC cluster job
scripts.

Honesty note
------------
This environment has no licensed ANSYS Fluent installation (``fluent`` is
not on PATH here -- confirmed via ``shutil.which`` at import/use time, not
assumed). This module has therefore only been exercised, and can only be
tested in this repository, along two paths:

  1. Journal-text generation (``journal_builder.build_journal``) --
     fully real logic, fully testable with no Fluent install.
  2. The "executable not found" error path here -- also fully real and
     testable, and it is the actual path a caller hits in this
     environment.

The subprocess-invocation and log-capture code itself is straightforward
and follows the same pattern as the OpenFOAM runner (which *is* exercised
against a real OpenFOAM install elsewhere in this repo's test suite), but
the specific claim "fluent 3ddp -g -i journal.jou correctly drives a real
solve end-to-end" is NOT verified locally -- consistent with this
repository's existing convention for external tools that aren't installed
here (see e.g. ``pinneapple_design/geometry/io/iges.py`` for gmsh, or
``cfd_formats/abaqus_reader.py``'s ``.odb`` bridge for Abaqus).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .journal_builder import FluentCaseConfig, build_journal


@dataclass(frozen=True)
class FluentRunConfig:
    """Mirrors ``openfoam.runner.OpenFOAMRunConfig``'s role: how to invoke
    the external binary, not what to simulate (that's ``FluentCaseConfig``)."""

    executable: str = "fluent"          # name or full path of the fluent launcher
    precision: str = "3ddp"             # "2d" | "3d" | "2ddp" | "3ddp"
    n_procs: int = 1                    # -> "-t<n_procs>" if > 1
    work_dir: Optional[str] = None      # defaults to a fresh temp dir
    timeout_s: Optional[float] = None
    extra_args: Sequence[str] = field(default_factory=tuple)


@dataclass
class FluentRunResult:
    success: bool                        # fluent exited 0 AND output_case_file exists
    upd_result: object                   # PhysicalSample from fluent_mesh_to_upd, or None
    exit_code: int
    wall_time_s: float
    stdout_log_path: str
    stderr_log_path: str
    journal_path: str
    read_error: Optional[str] = None     # set if the run succeeded but reading the output failed


def _find_fluent_executable(executable: str) -> str:
    """Locate the ``fluent`` launcher on PATH (or validate an explicit
    path), raising a clear, actionable error if it isn't available --
    mirrors ``openfoam.runner``'s "don't silently no-op" behavior for a
    missing external tool."""
    candidate = Path(executable)
    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.exists() and candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(
            f"ANSYS Fluent executable path '{executable}' does not exist. Pass a valid path via "
            "FluentRunConfig(executable=...), or a bare name (e.g. \"fluent\") to search PATH."
        )

    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"ANSYS Fluent executable '{executable}' was not found on PATH. This machine has no "
            "licensed ANSYS Fluent installation available (or its 'bin' directory isn't on PATH). "
            "Install ANSYS Fluent and ensure the directory containing 'fluent' is on PATH, or pass "
            "an explicit path via FluentRunConfig(executable=\"/full/path/to/fluent\")."
        )
    return resolved


def run_fluent_case(
    case_config: FluentCaseConfig,
    run_config: Optional[FluentRunConfig] = None,
) -> FluentRunResult:
    """Build a journal from ``case_config``, launch a real Fluent batch
    process against it, and package the resulting case/data file via the
    existing ``fluent_mesh_to_upd`` bridge (imported, not reimplemented).

    Raises ``FileNotFoundError`` immediately (before touching the
    filesystem beyond validating ``case_config``) if the ``fluent``
    executable can't be located -- this never silently no-ops.
    """
    run_config = run_config or FluentRunConfig()
    case_config.validate()

    fluent_bin = _find_fluent_executable(run_config.executable)

    if run_config.work_dir:
        work_dir = Path(run_config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="pinneapple_fluent_"))

    journal_text = build_journal(case_config)
    journal_path = work_dir / "run.jou"
    journal_path.write_text(journal_text)

    stdout_log_path = work_dir / "fluent_stdout.log"
    stderr_log_path = work_dir / "fluent_stderr.log"

    cmd = [fluent_bin, run_config.precision, "-g", "-i", str(journal_path)]
    if run_config.n_procs > 1:
        cmd.append(f"-t{run_config.n_procs}")
    cmd.extend(run_config.extra_args)

    t0 = time.time()
    with open(stdout_log_path, "w") as out_f, open(stderr_log_path, "w") as err_f:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=out_f,
            stderr=err_f,
            timeout=run_config.timeout_s,
        )
    wall_time_s = time.time() - t0

    exit_code = proc.returncode
    output_exists = Path(case_config.output_case_file).exists()
    success = exit_code == 0 and output_exists

    upd_result = None
    read_error = None
    if success:
        try:
            # Reuse the existing Fluent-format reader -- do not reimplement
            # Fluent file parsing here. Fluent case files use the same
            # section-numbered TGrid/Gambit grammar as the .msh files that
            # reader targets (see fluent_mesh_reader module docstring),
            # which is why this reuse is valid rather than coincidental --
            # provided the case was written as ASCII (this module requests
            # that via "/file/binary-files no" by default; see
            # FluentCaseConfig.write_ascii).
            from ..cfd_formats.fluent_mesh_reader import fluent_mesh_to_upd

            upd_result = fluent_mesh_to_upd(case_config.output_case_file)
        except Exception as e:  # noqa: BLE001 - degrade to a reported error, not a crash
            read_error = f"{type(e).__name__}: {e}"

    return FluentRunResult(
        success=success,
        upd_result=upd_result,
        exit_code=exit_code,
        wall_time_s=wall_time_s,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        journal_path=str(journal_path),
        read_error=read_error,
    )
