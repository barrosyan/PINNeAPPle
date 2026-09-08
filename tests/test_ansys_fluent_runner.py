"""Tests for the ANSYS Fluent live-solver coupling
(``pinneapple_simulation/external_solvers/ansys/``).

Journal-text generation is fully real logic and is tested thoroughly here
with no Fluent installation required. The actual subprocess-launch path
can only be exercised for its error handling in this environment: this
machine has no licensed ANSYS Fluent installation (confirmed below via
``shutil.which("fluent") is None`` rather than assumed), so
``run_fluent_case`` is only tested along the "executable not found" path.
If a real ``fluent`` binary is ever available, the guarded test at the
bottom of this file will additionally attempt (and report on) a minimal
real run.
"""
from __future__ import annotations

import shutil

import pytest

from pinneapple_simulation.external_solvers.ansys.journal_builder import (
    FluentCaseConfig,
    build_journal,
)
from pinneapple_simulation.external_solvers.ansys.runner import (
    FluentRunConfig,
    run_fluent_case,
)

FLUENT_AVAILABLE = shutil.which("fluent") is not None


# ---------------------------------------------------------------------------
# journal_builder.build_journal
# ---------------------------------------------------------------------------

def test_build_journal_steady_minimal_case_file():
    cfg = FluentCaseConfig(
        case_file="/data/inlet_flow.cas.h5",
        output_case_file="/data/out/inlet_flow_solved.cas",
        steady=True,
        iterations=250,
        initialize=True,
    )
    text = build_journal(cfg)
    lines = text.splitlines()

    assert lines[0] == '/file/read-case "/data/inlet_flow.cas.h5"'
    # ASCII-forcing toggle defaults on
    assert "/file/binary-files" in lines
    bin_idx = lines.index("/file/binary-files")
    assert lines[bin_idx + 1] == "no"
    assert "/solve/initialize/initialize-flow" in lines
    assert "/solve/iterate 250" in lines
    assert '/file/write-case-data "/data/out/inlet_flow_solved.cas"' in lines
    assert lines[-2] == "exit"
    assert lines[-1] == "yes"

    # Ordering: read-case must come before iterate, iterate before write-case-data,
    # write-case-data before exit.
    read_idx = lines.index('/file/read-case "/data/inlet_flow.cas.h5"')
    iterate_idx = lines.index("/solve/iterate 250")
    write_idx = lines.index('/file/write-case-data "/data/out/inlet_flow_solved.cas"')
    exit_idx = lines.index("exit")
    assert read_idx < iterate_idx < write_idx < exit_idx

    # No BC lines and no dual-time-iterate for this steady, BC-free config.
    assert not any("boundary-conditions" in l for l in lines)
    assert not any("dual-time-iterate" in l for l in lines)


def test_build_journal_transient_uses_dual_time_iterate():
    cfg = FluentCaseConfig(
        case_file="/data/pulsatile.cas.h5",
        output_case_file="/data/out/pulsatile_solved.cas",
        steady=False,
        time_steps=40,
        max_iterations_per_time_step=15,
    )
    text = build_journal(cfg)
    lines = text.splitlines()

    assert "/solve/dual-time-iterate 40 15" in lines
    assert not any(l.startswith("/solve/iterate ") for l in lines)


def test_build_journal_mesh_import_when_no_case_file():
    cfg = FluentCaseConfig(
        mesh_file="/data/duct.msh",
        output_case_file="/data/out/duct_solved.cas",
    )
    text = build_journal(cfg)
    lines = text.splitlines()
    assert lines[0] == '/file/import/fluent-mesh "/data/duct.msh"'
    assert not any(l.startswith("/file/read-case") for l in lines)


def test_build_journal_boundary_condition_overrides():
    cfg = FluentCaseConfig(
        case_file="/data/pipe.cas.h5",
        output_case_file="/data/out/pipe_solved.cas",
        velocity_inlet_bcs={"inlet": 12.5},
        pressure_outlet_bcs={"outlet": 0.0, "vent": 101325.0},
    )
    text = build_journal(cfg)
    lines = text.splitlines()

    assert "/define/boundary-conditions/set/velocity-inlet inlet () vmag no 12.5 quit" in lines
    assert "/define/boundary-conditions/set/pressure-outlet outlet () gauge-pressure no 0.0 quit" in lines
    assert "/define/boundary-conditions/set/pressure-outlet vent () gauge-pressure no 101325.0 quit" in lines

    # BCs must be applied after the case is read and before the solve.
    read_idx = lines.index('/file/read-case "/data/pipe.cas.h5"')
    iterate_idx = lines.index("/solve/iterate 100")
    inlet_idx = lines.index("/define/boundary-conditions/set/velocity-inlet inlet () vmag no 12.5 quit")
    assert read_idx < inlet_idx < iterate_idx


def test_build_journal_no_bc_overrides_produces_no_bc_lines():
    cfg = FluentCaseConfig(
        case_file="/data/plain.cas.h5",
        output_case_file="/data/out/plain_solved.cas",
    )
    text = build_journal(cfg)
    assert "boundary-conditions" not in text


def test_build_journal_residual_convergence_applies_uniform_value():
    cfg = FluentCaseConfig(
        case_file="/data/lam.cas.h5",
        output_case_file="/data/out/lam_solved.cas",
        residual_convergence=1e-4,
    )
    text = build_journal(cfg)
    assert "/solve/monitors/residual/convergence-criteria 0.0001 0.0001 0.0001 0.0001" in text.splitlines()


def test_build_journal_write_ascii_false_skips_binary_toggle():
    cfg = FluentCaseConfig(
        case_file="/data/x.cas.h5",
        output_case_file="/data/out/x_solved.cas",
        write_ascii=False,
    )
    text = build_journal(cfg)
    assert "/file/binary-files" not in text


def test_fluent_case_config_requires_case_or_mesh():
    cfg = FluentCaseConfig(output_case_file="/data/out/x.cas")
    with pytest.raises(ValueError):
        build_journal(cfg)


def test_fluent_case_config_rejects_both_case_and_mesh():
    cfg = FluentCaseConfig(
        case_file="/data/a.cas.h5",
        mesh_file="/data/a.msh",
        output_case_file="/data/out/a_solved.cas",
    )
    with pytest.raises(ValueError):
        build_journal(cfg)


def test_fluent_case_config_rejects_nonpositive_iterations():
    cfg = FluentCaseConfig(
        case_file="/data/a.cas.h5",
        output_case_file="/data/out/a_solved.cas",
        iterations=0,
    )
    with pytest.raises(ValueError):
        build_journal(cfg)


# ---------------------------------------------------------------------------
# runner.run_fluent_case -- error path (no Fluent installed here)
# ---------------------------------------------------------------------------

def test_fluent_not_on_path_in_this_environment():
    """Documents the actual state of this environment: no licensed ANSYS
    Fluent install. If this ever starts failing, the guarded real-run test
    below will start actually exercising a live solve instead of the
    error path."""
    assert shutil.which("fluent") is None


def test_run_fluent_case_raises_clear_error_when_executable_missing():
    cfg = FluentCaseConfig(
        case_file="/data/does_not_matter.cas.h5",
        output_case_file="/tmp/does_not_matter_solved.cas",
    )
    run_cfg = FluentRunConfig(executable="definitely_not_a_real_fluent_binary_xyz")
    with pytest.raises(FileNotFoundError, match="not found on PATH"):
        run_fluent_case(cfg, run_cfg)


def test_run_fluent_case_raises_clear_error_for_bad_explicit_path():
    cfg = FluentCaseConfig(
        case_file="/data/does_not_matter.cas.h5",
        output_case_file="/tmp/does_not_matter_solved.cas",
    )
    run_cfg = FluentRunConfig(executable="/nonexistent/path/to/fluent")
    with pytest.raises(FileNotFoundError):
        run_fluent_case(cfg, run_cfg)


def test_run_fluent_case_does_not_silently_no_op(tmp_path):
    """The executable-not-found error must be raised before any journal
    file is written -- i.e. this never silently no-ops and returns some
    empty/placeholder result."""
    cfg = FluentCaseConfig(
        case_file="/data/x.cas.h5",
        output_case_file=str(tmp_path / "out.cas"),
    )
    run_cfg = FluentRunConfig(executable="definitely_not_a_real_fluent_binary_xyz", work_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        run_fluent_case(cfg, run_cfg)
    assert not (tmp_path / "run.jou").exists()


@pytest.mark.skipif(not FLUENT_AVAILABLE, reason="No real ANSYS Fluent installation available in this environment")
def test_run_fluent_case_real_minimal_run(tmp_path):
    """Only runs if a real 'fluent' binary is actually on PATH somewhere.
    In that (currently untrue-here) case, attempt the smallest possible
    real batch invocation and report what happened rather than asserting
    a specific outcome, since we have no known-good case file bundled in
    this repo to point it at."""
    cfg = FluentCaseConfig(
        mesh_file="tests/fixtures/cfd_formats/real_meshio_ansys.msh",
        output_case_file=str(tmp_path / "real_run_out.cas"),
        iterations=1,
    )
    result = run_fluent_case(cfg, FluentRunConfig(work_dir=str(tmp_path), timeout_s=120))
    # No strict assertion on `success` -- report-only, since a real Fluent
    # run's success depends on licensing/case validity we can't control here.
    assert result.journal_path
