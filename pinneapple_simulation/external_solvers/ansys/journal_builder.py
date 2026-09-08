"""ANSYS Fluent batch-mode journal (``.jou``) generation.

Fluent has no Python/REST scripting surface used here -- like every other
"drive a real external solver" integration in this repository
(``external_solvers/openfoam/runner.py`` shells out to real OpenFOAM
executables), the only way to actually run it non-interactively is its
documented Text User Interface (TUI), invoked in batch mode with a journal
file of TUI commands (``fluent 3ddp -g -i run.jou``). The TUI command set
used below (``/file/read-case``, ``/solve/initialize/initialize-flow``,
``/solve/iterate``, ``/solve/dual-time-iterate``,
``/file/write-case-data``, ``/file/binary-files``, ``exit``) is Fluent's
own public, documented command-line interface -- the same commands that
appear throughout ANSYS's own Fluent Text Command List documentation and
in essentially every publicly published Fluent journal-file tutorial. This
is knowledge of a standard tool's documented CLI (like knowing ``git
commit -m``), not anything reverse-engineered or proprietary.

Confidence level, command by command
-------------------------------------
High confidence (these are the commands used by virtually every public
Fluent journal example and have been stable across Fluent versions for
decades):
  - ``/file/read-case <path>``
  - ``/file/binary-files no`` (ASCII/text I/O toggle)
  - ``/solve/initialize/initialize-flow``
  - ``/solve/iterate <n>``
  - ``/file/write-case-data <path>``
  - ``exit`` / ``yes``

Medium confidence (documented, but with more version-to-version drift in
exact prompt sequencing than the commands above):
  - ``/solve/dual-time-iterate <n-time-steps> <max-iterations-per-step>``
    for transient runs.
  - ``/solve/monitors/residual/convergence-criteria <values...>`` -- the
    number and order of prompted values depends on which equations
    (turbulence, energy, species, ...) are active in the case, which this
    module cannot know without reading the case file. See
    ``FluentCaseConfig.residual_convergence`` docstring for the narrow,
    explicit assumption made here (a fixed 4-slot laminar flow/continuity
    + 3 velocity components ordering).
  - ``/define/boundary-conditions/set/<zone-type> <zone> () <field> no
    <value> quit`` -- this "set" shorthand (zone list, empty parens for
    "no additional zones", field keyword, "no" to skip profile/expression
    input, a literal value, ``quit`` to close the per-zone menu) is the
    idiom used in numerous public Fluent journal tutorials for setting a
    single scalar BC value without walking the full interactive panel.
    Deliberately limited to exactly two BC kinds for this reason -- see
    below.

Deliberately NOT implemented (left out due to insufficient confidence,
per this module's honesty policy -- do not guess at TUI syntax you are not
sure of):
  - Any boundary condition beyond velocity-inlet velocity-magnitude and
    pressure-outlet gauge-pressure (e.g. mass-flow-inlet, wall
    temperature/heat-flux, turbulence intensity/length-scale, species
    fractions, multiphase BCs). These involve longer, model-dependent
    prompt sequences this module cannot generate reliably without a real
    Fluent session to verify against.
  - Turbulence/physics model selection (``/define/models/...``) -- the
    input case file is assumed to already have the desired model set up.
  - Mesh quality/adaption commands.
  - Parallel partitioning TUI commands (processor count is a Fluent
    *launch* flag, ``-t<n>``, handled by ``ansys.runner``, not a journal
    command).

None of this has been exercised against a real Fluent installation (see
``ansys/runner.py`` and ``tests/test_ansys_fluent_runner.py`` for why:
this environment has no licensed ``fluent`` binary). Only the generated
journal *text* is tested here, against the documented command syntax
above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


# Fixed prompt order assumed for the (medium-confidence, laminar-only)
# convergence-criteria shortcut -- see FluentCaseConfig.residual_convergence.
_DEFAULT_RESIDUAL_NAMES: Sequence[str] = (
    "continuity", "x-velocity", "y-velocity", "z-velocity",
)


@dataclass(frozen=True)
class FluentCaseConfig:
    """Configuration for one basic batch Fluent run.

    Exactly one of ``case_file`` / ``mesh_file`` must be given: an existing
    Fluent case (``.cas``/``.cas.h5``, read via ``/file/read-case``) or a
    raw mesh to import (via ``/file/import/fluent-mesh``) when no case has
    been set up yet. A case file is strongly preferred -- it already
    carries model/BC-type setup that a bare mesh does not.
    """

    output_case_file: str                     # where /file/write-case-data writes the result
    case_file: Optional[str] = None            # existing .cas/.cas.h5 to read
    mesh_file: Optional[str] = None            # raw mesh to import if no case_file

    steady: bool = True
    iterations: int = 100                      # steady: /solve/iterate <n>
    time_steps: int = 20                       # transient: number of time steps
    max_iterations_per_time_step: int = 20      # transient: sub-iterations per step
    initialize: bool = True                     # run hybrid initialization before iterating

    # Simple name -> scalar-value boundary condition overrides. Deliberately
    # narrow -- see module docstring for why only these two BC kinds are
    # covered. Keys are the zone names as they exist in the case/mesh.
    velocity_inlet_bcs: Dict[str, float] = field(default_factory=dict)   # zone -> velocity magnitude [m/s]
    pressure_outlet_bcs: Dict[str, float] = field(default_factory=dict)  # zone -> gauge pressure [Pa]

    # Optional single scalar residual target applied uniformly across a
    # fixed, laminar-flow 4-equation set (continuity + 3 velocity
    # components). This is a narrow, explicit simplification -- a case
    # with energy/turbulence/species active has more residuals than this
    # covers, and Fluent's actual prompt count/order for those depends on
    # which models are active (see module docstring "medium confidence").
    # Leave as None (the default) for any non-trivial physics case.
    residual_convergence: Optional[float] = None

    # Our downstream reader (cfd_formats.fluent_mesh_reader) only supports
    # ASCII-encoded Fluent files, so by default we force ASCII output via
    # "/file/binary-files no" before writing. Only set this False if you
    # will read the output some other way.
    write_ascii: bool = True

    def validate(self) -> None:
        if not self.case_file and not self.mesh_file:
            raise ValueError(
                "FluentCaseConfig needs either case_file (existing .cas/.cas.h5) or "
                "mesh_file (a mesh to import) -- got neither."
            )
        if self.case_file and self.mesh_file:
            raise ValueError(
                "FluentCaseConfig got both case_file and mesh_file -- provide exactly one."
            )
        if not self.output_case_file:
            raise ValueError("FluentCaseConfig.output_case_file is required.")
        if not self.steady and self.time_steps <= 0:
            raise ValueError("time_steps must be > 0 for a transient run.")
        if self.steady and self.iterations <= 0:
            raise ValueError("iterations must be > 0 for a steady run.")


def build_journal(config: FluentCaseConfig) -> str:
    """Generate Fluent TUI journal-file text for ``config``.

    Command order: read case/mesh -> (optional) force ASCII output ->
    boundary-condition overrides -> (optional) convergence criteria ->
    (optional) flow initialization -> solve (steady iterate or transient
    dual-time-iterate) -> write case+data -> exit.
    """
    config.validate()
    lines = []

    if config.case_file:
        lines.append(f'/file/read-case "{config.case_file}"')
    else:
        lines.append(f'/file/import/fluent-mesh "{config.mesh_file}"')

    if config.write_ascii:
        lines.append("/file/binary-files")
        lines.append("no")

    for zone, vmag in config.velocity_inlet_bcs.items():
        lines.append(f"/define/boundary-conditions/set/velocity-inlet {zone} () vmag no {vmag} quit")

    for zone, pgauge in config.pressure_outlet_bcs.items():
        lines.append(f"/define/boundary-conditions/set/pressure-outlet {zone} () gauge-pressure no {pgauge} quit")

    if config.residual_convergence is not None:
        values = " ".join(str(config.residual_convergence) for _ in _DEFAULT_RESIDUAL_NAMES)
        lines.append(f"/solve/monitors/residual/convergence-criteria {values}")

    if config.initialize:
        lines.append("/solve/initialize/initialize-flow")

    if config.steady:
        lines.append(f"/solve/iterate {config.iterations}")
    else:
        lines.append(f"/solve/dual-time-iterate {config.time_steps} {config.max_iterations_per_time_step}")

    lines.append(f'/file/write-case-data "{config.output_case_file}"')
    lines.append("exit")
    lines.append("yes")

    return "\n".join(lines) + "\n"
