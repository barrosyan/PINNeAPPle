"""Subprocess-based MATLAB runner.

Use this when the MATLAB Engine for Python is unavailable (e.g. CI, Docker).
Communication happens via .mat files written to disk.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def run_matlab_script(
    script_path: Union[str, Path],
    *,
    matlab_executable: str = "matlab",
    timeout: int = 300,
    extra_flags: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run a MATLAB script via subprocess (-batch mode).

    Parameters
    ----------
    script_path : path to the .m script
    matlab_executable : path/name of the matlab binary
    timeout : seconds before the subprocess is killed
    extra_flags : additional CLI flags passed after -batch

    Returns
    -------
    CompletedProcess (stdout/stderr captured)
    """
    p = Path(script_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    batch_cmd = f"addpath('{p.parent}'); {p.stem}"
    flags = ["-batch", batch_cmd] + (extra_flags or [])
    return subprocess.run(
        [matlab_executable, "-nodesktop", "-nosplash"] + flags,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def run_matlab_function(
    func_name: str,
    args: List[Any],
    out_mat_path: Union[str, Path],
    *,
    matlab_executable: str = "matlab",
    timeout: int = 300,
) -> None:
    """Call a MATLAB function via a generated wrapper script.

    Saves function arguments to a temp .mat file, calls the function from
    MATLAB, and writes the result to *out_mat_path*.

    Parameters
    ----------
    func_name : name of the MATLAB function (must be on MATLAB path)
    args : list of numpy arrays / scalars to pass as positional arguments
    out_mat_path : where MATLAB should write the result (.mat)
    """
    from .mat_io import save_mat

    with tempfile.TemporaryDirectory() as tmpdir:
        args_path = Path(tmpdir) / "args.mat"
        save_mat(args_path, {"args": args})

        wrapper = (
            f"load('{args_path}');\n"
            f"result = {func_name}(args{{:}});\n"
            f"save('{out_mat_path}', 'result');\n"
        )
        wrapper_path = Path(tmpdir) / "_wrapper.m"
        wrapper_path.write_text(wrapper)

        subprocess.run(
            [matlab_executable, "-nodesktop", "-nosplash",
             "-batch", f"run('{wrapper_path}')"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
