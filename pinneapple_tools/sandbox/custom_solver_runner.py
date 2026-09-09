"""Run an untrusted, user-supplied Python solver script out-of-process,
with best-effort resource limits.

SECURITY WARNING -- read this before using this module for anything
------------------------------------------------------------------------
``run_custom_solver`` isolates a script into a *separate OS process* and
applies POSIX resource limits (``resource.setrlimit``) plus a wall-clock
``subprocess.run(..., timeout=...)``. That is **process-level resource
limiting, not a sandbox**, and it does NOT provide a security boundary:

  * No filesystem restriction. The script can read, write, or delete any
    file the calling OS user can touch.
  * No network restriction. The script can open sockets, make HTTP
    requests, exfiltrate data, etc.
  * No syscall/capability restriction of any kind (no seccomp, no
    namespaces, no chroot). It can spawn further subprocesses, load
    arbitrary shared libraries, etc.
  * ``RLIMIT_AS`` bounds *virtual address space*, not resident physical
    memory -- it stops a script that tries to allocate a huge array, but
    is not a general memory-abuse defense (e.g. many small long-lived
    allocations that never trip a single large allocation), and some
    platforms/allocators don't enforce it as tightly as Linux glibc does.
  * ``RLIMIT_CPU`` bounds *CPU time*, not wall-clock time -- a script that
    sleeps, blocks on I/O, or spins a busy network wait can run
    indefinitely without touching its CPU budget. That's why this module
    ALSO enforces the wall-clock ``timeout`` via ``subprocess.run``, which
    is the limit that actually catches an infinite ``while True: pass``
    loop doing real (CPU-bound) work, and the wall-clock timeout is the
    one to rely on for a hang of any kind.

Use this only for scripts you already have some trust in (e.g. generated
by ``pinneapple_physics.codegen`` from a spec you control, or a
cooperative user's code) as a safety net against bugs -- runaway loops,
accidental huge allocations -- never as the only defense against a
genuinely adversarial, untrusted script. For real isolation, put a proper
sandbox in front of this: a container with ``--network=none`` and a
read-only filesystem, or a stronger technology (gVisor, Firecracker,
nsjail). This module does not provide that itself.

Platform support
-----------------
POSIX only (Linux, macOS) -- it relies on the stdlib ``resource`` module
and ``subprocess``'s ``preexec_fn`` (fork-based), neither of which exists
on Windows. ``run_custom_solver`` raises ``RuntimeError`` up front on an
unsupported platform rather than silently running the script with no
limits at all.

Calling convention
--------------------
``script_path`` must define a module-level ``solve(params: dict) -> dict``
function whose returned dict contains at least a ``"coords"`` key -- the
same convention used by every script
``pinneapple_physics.codegen.fdm_script_generator``/
``fenics_script_generator`` produces, so those generated scripts (or any
hand-written script following the same convention) can be run here
unmodified.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional

try:
    import resource  # POSIX only
    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover -- exercised only on Windows
    resource = None  # type: ignore[assignment]
    _HAS_RESOURCE = False


# ---------------------------------------------------------------------------
# npz flatten/unflatten helpers (shared by the child driver and the parent)
# ---------------------------------------------------------------------------
#
# solve(params) results are dicts that may nest one level deep (e.g.
# {"coords": {"x": arr, "t": arr}, "u": arr, "params": {"nu": 0.05}}), but
# np.savez only stores a flat mapping of name -> array. We flatten nested
# dicts with a "/"-joined key and reverse that on load, so the caller gets
# back the same nested shape ``solve()`` produced.

def _flatten_for_npz(value: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            flat.update(_flatten_for_npz(v, f"{prefix}{k}/"))
    else:
        flat[prefix.rstrip("/")] = value
    return flat


def _to_npz_array(value: Any):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return np.array(str(value))


def _unflatten_from_npz(npz: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in npz.files:
        parts = key.split("/")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        val = npz[key]
        if val.ndim == 0:
            val = val.item()
        node[parts[-1]] = val
    return result


# ---------------------------------------------------------------------------
# Child-process driver (invoked as `python -m pinneapple_tools.sandbox
# .custom_solver_runner`, i.e. this module's own __main__ block below)
# ---------------------------------------------------------------------------

def _child_main(argv: Optional[list] = None) -> None:
    """Runs INSIDE the resource-limited subprocess: load the target
    script, call its solve(params), and report the outcome via a status
    JSON file (never via a raised exception escaping this process, so the
    parent can always tell success from failure instead of just seeing a
    nonzero exit code)."""
    import argparse
    import runpy
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--status-json", required=True)
    args = ap.parse_args(argv)

    status: Dict[str, Any] = {"ok": False}
    try:
        params = json.loads(args.params_json)
        namespace = runpy.run_path(args.script, run_name="__pinneapple_custom_solver__")
        solve_fn = namespace.get("solve")
        if not callable(solve_fn):
            raise ValueError(f"{args.script!r} does not define a callable solve(params)")

        result = solve_fn(params)
        if not isinstance(result, dict) or "coords" not in result:
            raise ValueError(
                "solve(params) must return a dict containing at least a 'coords' key; "
                f"got {type(result).__name__}"
                + (f" with keys {list(result.keys())}" if isinstance(result, dict) else "")
            )

        import numpy as np

        flat = {k: _to_npz_array(v) for k, v in _flatten_for_npz(result).items()}
        np.savez(args.out_npz, **flat)
        status = {"ok": True}
    except BaseException as exc:  # noqa: BLE001 -- deliberately broad: report, never swallow
        status = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open(args.status_json, "w") as fh:
        json.dump(status, fh)


# ---------------------------------------------------------------------------
# Parent-side API
# ---------------------------------------------------------------------------

def run_custom_solver(
    script_path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 60.0,
    memory_limit_mb: int = 512,
) -> Dict[str, Any]:
    """Run ``script_path``'s ``solve(params)`` in a separate, resource-
    limited subprocess. See the module docstring's SECURITY WARNING for
    exactly what protection this does (and does not) provide.

    Parameters
    ----------
    script_path : path to a Python file defining ``solve(params: dict) ->
        dict`` (the dict must contain at least a ``"coords"`` key) — e.g.
        anything produced by ``pinneapple_physics.codegen``.
    params : forwarded to ``solve()`` in the child process, JSON-encoded
        for the trip across the process boundary, so it must be
        JSON-serializable (plain numbers/strings/lists/dicts/bools).
    timeout : wall-clock seconds before the subprocess is killed. This is
        the limit that actually catches a hang of any kind (an infinite
        CPU-bound loop, a blocked I/O call, ...).
    memory_limit_mb : soft cap on the child's virtual address space
        (``RLIMIT_AS``), in megabytes. Bounds virtual memory, not resident
        physical memory -- see the module docstring.

    Returns
    -------
    On success: the ``solve()`` result dict (nested structure preserved),
    plus ``"success": True`` and ``"elapsed_s": float``.
    On failure (exception raised inside ``solve()``, malformed result,
    timeout, or the process being killed outright by the OS/resource
    limits): ``{"success": False, "error": <exception class name or
    "TimeoutExpired"/"ProcessTerminated">, "message": str, ...}`` -- the
    error is always surfaced here, never silently swallowed.
    """
    if not sys.platform.startswith(("linux", "darwin")):
        raise RuntimeError(
            "run_custom_solver relies on POSIX resource.setrlimit (RLIMIT_CPU, "
            "RLIMIT_AS) via subprocess preexec_fn, which only exists on "
            f"Linux/macOS, not {sys.platform!r}. Refusing to run with no limits "
            "applied rather than silently skipping them."
        )
    if not _HAS_RESOURCE:  # pragma: no cover -- unreachable given the check above
        raise RuntimeError("The stdlib 'resource' module is unavailable on this platform.")

    params = dict(params or {})
    mem_bytes = int(memory_limit_mb) * 1024 * 1024
    # CPU-time cap is intentionally more generous than the wall-clock
    # `timeout`: it's a backstop against CPU-bound runaway loops, not the
    # primary enforcement (that's `timeout` below, which also catches
    # blocking I/O / sleeps that never touch the CPU budget).
    cpu_seconds = max(1, int(timeout) + 5)

    def _apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            # Some platforms/process states refuse RLIMIT_AS outright
            # (observed to be flaky on macOS in particular); degrade to
            # CPU-limit-only rather than crash the whole call.
            pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_npz = os.path.join(tmp_dir, "result.npz")
        status_json = os.path.join(tmp_dir, "status.json")
        cmd = [
            sys.executable, "-m", "pinneapple_tools.sandbox.custom_solver_runner",
            "--script", os.path.abspath(script_path),
            "--params-json", json.dumps(params),
            "--out-npz", out_npz,
            "--status-json", status_json,
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                preexec_fn=_apply_limits,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "TimeoutExpired",
                "message": f"Solver did not finish within timeout={timeout}s (wall-clock).",
                "elapsed_s": time.monotonic() - start,
            }

        elapsed = time.monotonic() - start

        if os.path.exists(status_json):
            with open(status_json) as fh:
                status = json.load(fh)
        else:
            # The child never got to write its status file -- most likely
            # killed outright by the OS (OOM killer / a hard resource-limit
            # violation the interpreter couldn't catch as a Python
            # exception) rather than a reportable Python-level failure.
            status = {
                "ok": False,
                "error_type": "ProcessTerminated",
                "error_message": (
                    f"Subprocess exited with code {proc.returncode} without "
                    "writing a status file -- likely killed by the OS "
                    "(out-of-memory or a hard resource-limit violation) "
                    "rather than raising a catchable Python exception."
                ),
            }

        if not status.get("ok"):
            return {
                "success": False,
                "error": status.get("error_type", "UnknownError"),
                "message": status.get("error_message", ""),
                "traceback": status.get("traceback", ""),
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:] if proc.stderr else "",
                "elapsed_s": elapsed,
            }

        import numpy as np

        with np.load(out_npz, allow_pickle=False) as npz:
            result = _unflatten_from_npz(npz)
        result["success"] = True
        result["elapsed_s"] = elapsed
        return result


if __name__ == "__main__":
    _child_main()
