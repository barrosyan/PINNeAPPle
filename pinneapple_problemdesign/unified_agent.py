"""Compose ``DesignAgent`` and ``PhysicsOrchestrator`` into one working loop.

This repo has two independent, real, separately-tested agent/orchestration
systems:

* :class:`pinneapple_problemdesign.agent.DesignAgent` — natural-language
  problem *elicitation*. It is staged via :class:`~.state.DesignState`
  and produces a :class:`~.schema.DesignReport` (spec + gaps + plan) once
  enough information has been gathered.
* :class:`pinneapple_worldmodel.orchestrator.PhysicsOrchestrator` — given a
  :class:`~pinneapple_worldmodel.orchestrator.ProblemStatement`, selects and
  *executes* real tools from
  :class:`~pinneapple_worldmodel.physics_tools.PhysicsToolRegistry` to
  actually accomplish a physics goal.

Design decision: composition, not a class-hierarchy merge
-----------------------------------------------------------
Both systems are live and used elsewhere in this repo (``DesignAgent`` by
the problem-design CLI/examples, ``PhysicsOrchestrator`` directly by
downstream physics workflows). Rewriting their internals into one merged
class hierarchy would risk breaking either system for its existing callers,
for a purely cosmetic gain. ``pinneapple_problemdesign.knowledge.mapping``
already established the pattern this module follows: reach into the other
system read-only/compositionally, never edit it, never assume it must be
present.

:class:`UnifiedPhysicsAgent` in this module goes one step further than that
existing *read-only* bridge (which only lists tool *names* inside a plan's
text): it actually closes the functional loop —

    plain-English problem description
        -> DesignAgent elicitation (real, unmodified)
        -> ProblemSpec / DesignReport (real, unmodified)
        -> ProblemStatement (new, honest field mapping -- see below)
        -> PhysicsOrchestrator.solve() (real, unmodified)
        -> OrchestratorResult (real tool output)

without changing a single line of ``pinneapple_problemdesign.agent``,
``pinneapple_worldmodel.orchestrator``, or ``pinneapple_worldmodel.physics_tools``.
This delivers the same user-facing value a literal merge would ("a user
never has to manually bridge the two systems again") with none of the risk
of rewriting two live class hierarchies.

Field mapping: ``ProblemSpec`` -> ``ProblemStatement``
-------------------------------------------------------
``to_problem_statement`` is an honest, field-by-field translation derived
from actually reading both dataclasses (see ``orchestrator.py`` and
``schema.py``), not a guess. Summary (see the method's own docstring/inline
comments for the full reasoning):

============================  ==========================  ============================================
ProblemSpec field             ProblemStatement field       Rule
============================  ==========================  ============================================
task_type                     kind                          Small, confident lookup table only
                                                             (forecasting->forecast,
                                                             inverse_problem->inverse,
                                                             pde_solution->forward,
                                                             optimization->design). Anything else
                                                             (neural_operator/control/anomaly_detection/
                                                             other) has no unambiguous orchestrator
                                                             analogue -- raises ``ValueError`` naming the
                                                             gap unless the caller passes ``kind=`` to
                                                             override explicitly.
goal (fallback: title)        description                  Direct copy; ``description`` is documented
                                                             as free text used for PDE auto-discovery,
                                                             and ``goal`` is literally that.
geometry.domain                domain_hint                  Lower-cased/underscored free text, or the
                                                             ProblemStatement default ``"unit_square"``
                                                             if empty. Verified (by reading
                                                             orchestrator.py in full) that no
                                                             ``_plan_*`` method ever reads
                                                             ``domain_hint`` -- it is inert metadata
                                                             today, so a lossy slug is safe: it cannot
                                                             change which tools run.
validation.primary_metrics /  output                        Base ``["model"]`` (ProblemStatement's own
validation.acceptance_criteria                              default) plus ``"validation_report"`` when
                                                             the spec actually captured validation
                                                             intent. Additive only.
(none)                        pde_hint, scenarios           **Not derived.** ``pde_hint``/``scenarios``
                                                             speak pinneapple_worldmodel's own
                                                             scenario-name vocabulary (``"heat_2d"``,
                                                             ``"ns2d_cavity"``, ...). ``ProblemSpec``
                                                             (and the PINN pde-identification vocabulary
                                                             used by ``codegen.build_pinneapple_spec``,
                                                             e.g. ``"heat_equation"``) never expresses a
                                                             value in that vocabulary. Bridging the two
                                                             would require guessing spatial
                                                             dimensionality / boundary conditions / IC
                                                             type that ``ProblemSpec`` does not capture
                                                             -- exactly what the non-invention policy
                                                             (``policy.py``) forbids. Left at
                                                             ``ProblemStatement``'s own defaults; pass
                                                             them via keyword overrides if you know the
                                                             right scenario.
(none)                         observations                 No source: ``ProblemSpec`` carries only
                                                             data *metadata* (``DataSpec``), never an
                                                             actual observation payload. Left ``None``
                                                             unless passed explicitly; if the resolved
                                                             ``kind`` is ``"inverse"`` (which
                                                             ``PhysicsOrchestrator._plan_inverse`` hard
                                                             -requires observations for) and none was
                                                             given, raises ``ValueError`` naming exactly
                                                             what is missing rather than silently passing
                                                             ``None`` through to a less-actionable failure
                                                             deep inside the orchestrator.
(none)                         params, model, n_samples,    No natural source field exists in
                               n_steps, device, save_dir,   ``ProblemSpec`` today (confirmed against the
                               verbose, extra               real dataclass read in schema.py -- e.g.
                                                             ``horizon``/``frequency`` are free-text
                                                             durations with no reliable unit-safe
                                                             conversion to an integer step count). Left
                                                             at ``ProblemStatement``'s own defaults;
                                                             override via keyword arguments to
                                                             ``to_problem_statement``/``run`` when known.
============================  ==========================  ============================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent import DesignAgent
from .protocol import LLMProvider
from .schema import ProblemSpec, DesignReport, Gap
from .state import DesignState
from .elicitation.stages import STAGES_ORDER

from pinneapple_worldmodel.orchestrator import (
    OrchestratorResult,
    PhysicsOrchestrator,
    ProblemStatement,
)


# ---------------------------------------------------------------------------
# task_type -> orchestrator kind
# ---------------------------------------------------------------------------
#
# Only mappings with a single, unambiguous orchestrator analogue are listed
# here (see the module docstring's field-mapping table for the reasoning).
# ``ProblemSpec.task_type`` values not listed (``neural_operator``,
# ``control``, ``anomaly_detection``, ``other``) have no confident match
# among PhysicsOrchestrator's eight ``kind`` values and must be supplied
# explicitly via ``kind=`` to avoid inventing a wrong one.
_TASK_TYPE_TO_KIND: Dict[str, str] = {
    "forecasting": "forecast",
    "inverse_problem": "inverse",
    "pde_solution": "forward",
    "optimization": "design",
}

# The full set of kinds PhysicsOrchestrator.solve() actually dispatches on
# (mirrors the `dispatch` dict read in orchestrator.py's solve()), used only
# to produce an actionable error message -- never guessed against.
_ORCHESTRATOR_KINDS = (
    "forward", "inverse", "design", "forecast",
    "uncertainty", "discovery", "digital_twin", "world_model",
)


@dataclass
class UnifiedAgentResult:
    """Bundled result of :meth:`UnifiedPhysicsAgent.run`.

    Attributes
    ----------
    status : str
        One of ``"executed"``, ``"needs_more_info"``, ``"execution_failed"``.
    design_report : DesignReport or None
        The real report produced by ``DesignAgent`` once elicitation
        completed. ``None`` when ``status == "needs_more_info"`` (no report
        exists yet -- ``DesignState.done`` is still ``False``).
    problem_statement : ProblemStatement or None
        The real, executable statement built by
        :meth:`UnifiedPhysicsAgent.to_problem_statement`. ``None`` unless
        elicitation completed.
    execution_result : OrchestratorResult or None
        ``PhysicsOrchestrator.solve()``'s real return value. ``None`` unless
        ``status == "executed"``.
    design_state : DesignState or None
        The live elicitation state (including any unresolved ``gaps``), so a
        caller can keep the conversation going with
        ``design_agent.ingest_user_message``/``.step`` when
        ``status == "needs_more_info"``.
    gaps : list of Gap
        The pending questions that blocked elicitation from completing.
        Populated only for ``status == "needs_more_info"``.
    questions_text : str
        Human-readable rendering of ``gaps`` (as produced by
        ``DesignAgent.step``'s own question-rewriting step).
    error : str or None
        ``f"{ExceptionType}: {message}"`` for a real, un-swallowed failure.
        Populated only for ``status == "execution_failed"``.
    """
    status: str
    design_report: Optional[DesignReport] = None
    problem_statement: Optional[ProblemStatement] = None
    execution_result: Optional[OrchestratorResult] = None
    design_state: Optional[DesignState] = None
    gaps: List[Gap] = field(default_factory=list)
    questions_text: str = ""
    error: Optional[str] = None


class UnifiedPhysicsAgent:
    """Compose a real ``DesignAgent`` with a real ``PhysicsOrchestrator``.

    This wraps a :class:`~.agent.DesignAgent` instance internally and never
    reimplements any elicitation logic -- every elicitation step is a direct
    delegation to the real ``DesignAgent`` methods (``start``,
    ``ingest_user_message``, ``step``). Execution is a direct delegation to
    a real ``PhysicsOrchestrator.solve()`` call. See the module docstring
    for the full design rationale and field-mapping table.

    Parameters
    ----------
    llm : LLMProvider
        Passed straight through to the internal ``DesignAgent``.
    use_orchestrator_bridge : bool
        Passed straight through to the internal ``DesignAgent`` (controls
        its own, separate, read-only "available orchestrator tools" plan
        annotation -- unrelated to, and compatible with, this class's
        actual execution step).
    orchestrator : PhysicsOrchestrator or None
        Use a specific orchestrator instance (e.g. with a custom
        ``PhysicsToolRegistry``). If ``None`` (default), a real
        ``PhysicsOrchestrator()`` is lazily constructed on first use.
    orchestrator_verbose : bool
        ``verbose`` flag for the lazily-constructed default orchestrator.
        Ignored if an explicit ``orchestrator`` is given.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        use_orchestrator_bridge: bool = True,
        orchestrator: Optional[PhysicsOrchestrator] = None,
        orchestrator_verbose: bool = True,
    ) -> None:
        self.design_agent = DesignAgent(llm, use_orchestrator_bridge=use_orchestrator_bridge)
        self._orchestrator = orchestrator
        self._orchestrator_verbose = orchestrator_verbose

    @property
    def orchestrator(self) -> PhysicsOrchestrator:
        """The real ``PhysicsOrchestrator`` this agent executes against.

        Constructed lazily (on first access) if none was supplied, so
        merely constructing a ``UnifiedPhysicsAgent`` never pays the cost of
        ``PhysicsToolRegistry().register_all()``.
        """
        if self._orchestrator is None:
            self._orchestrator = PhysicsOrchestrator(verbose=self._orchestrator_verbose)
        return self._orchestrator

    # ------------------------------------------------------------------
    # Thin delegation to the real DesignAgent (manual/multi-turn control)
    # ------------------------------------------------------------------

    def start(self) -> DesignState:
        """Delegates to ``DesignAgent.start()``."""
        return self.design_agent.start()

    def ingest(self, state: DesignState, user_text: str) -> None:
        """Delegates to ``DesignAgent.ingest_user_message()``."""
        self.design_agent.ingest_user_message(state, user_text)

    def step(self, state: DesignState) -> Dict[str, Any]:
        """Delegates to ``DesignAgent.step()``."""
        return self.design_agent.step(state)

    # ------------------------------------------------------------------
    # Elicitation driving (still pure delegation -- just loops step())
    # ------------------------------------------------------------------

    def drive_elicitation(
        self,
        user_text: str,
        *,
        initial_spec: Optional[ProblemSpec] = None,
        max_stage_advances: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Drive one starting message through ``DesignAgent`` as far as it
        will go *without further human input*.

        ``initial_spec``, if given, seeds ``DesignState.spec`` (a real,
        live ``ProblemSpec``) *before* ``user_text`` is ingested. This is a
        deliberate, additive escape hatch for a real, verified limitation
        of the live, unmodified ``merge_into_spec`` (``pinneapple_problemdesign
        /merge.py``, not part of this module and not edited by it): its
        ``_deep_merge`` only ever overwrites a top-level key when the
        destination is already a ``dict`` or an "empty" sentinel
        (``None``/``""``/``[]``/``{}``), or is a brand-new key -- so it can
        NEVER descend into ``ProblemSpec``'s nested dataclass sub-objects
        (``DataSpec``/``PhysicsSpec``/``GeometrySpec``/``ValidationSpec``/
        ``DeploymentSpec``/``ConstraintsSpec``), because
        ``spec.__dict__["data"]`` etc. is always a dataclass instance, never
        a plain ``dict``. Concretely: the ``data`` and ``validation``
        elicitation stages require exactly such nested fields
        (``data.sources``, ``validation.primary_metrics``, ...), and there
        is no way to fill them from chat text alone through
        ``DesignAgent``'s real, unmodified extraction+merge pipeline today
        -- confirmed by direct reproduction, and matching this repo's own
        ``examples/problem_designer/04_custom_provider_mock_end_to_end.py``,
        which also never reaches a finished report. ``initial_spec`` lets a
        caller who already has this structured information (a form, a
        config, a DB record -- the same spirit as
        ``examples/problem_designer/03_offline_spec_to_report.py``) supply
        it directly, while ``user_text`` still drives the free-text/flat
        fields (title/goal/task_type/domain_context/inputs/outputs/
        frequency/input_window/horizon) through the real extraction path.
        Fields already set on ``initial_spec`` are never overwritten by
        extraction (``merge_into_spec`` only fills currently-empty fields).

        This is honest about what a single-message, single-call convenience
        method can and cannot do: ``DesignAgent.step()`` advances at most one
        elicitation stage per call, and a stage only auto-advances when its
        ``required_fields`` are already satisfied (``elicitation/stages.py``
        gives ``physics``/``constraints``/``approach_selection``/
        ``finalization`` an *empty* ``required_fields``, so those always
        auto-advance and can still surface non-blocking ``nice_to_have``/
        ``important`` suggestion gaps in the same call's "questions" payload
        without actually being stuck).

        So the real, non-invented signal for "still making progress" is
        whether ``DesignState.stage`` actually changed on a given call --
        not whether the call happened to mention any gaps. This calls
        ``step()`` in a loop and keeps going as long as the stage keeps
        advancing; it stops the moment a call fails to advance the stage
        (i.e. that stage's real ``required_fields`` are genuinely unmet),
        returning whatever real questions ``DesignAgent`` surfaced. It never
        fabricates answers or forces a multi-turn conversation to finish.

        Returns
        -------
        dict with keys:
            ``"state"``  : the ``DesignState`` after driving.
            ``"result"`` : the last dict returned by ``DesignAgent.step()``
                           (either ``{"type": "report", ...}`` or
                           ``{"type": "questions", ...}``).
        """
        if max_stage_advances is None:
            # Headroom above the number of named stages: in the worst case
            # every stage needs its own step() call to advance, plus one
            # more call for the finalization step itself.
            max_stage_advances = len(STAGES_ORDER) + 2

        state = self.design_agent.start()
        if initial_spec is not None:
            state.spec = initial_spec
        self.design_agent.ingest_user_message(state, user_text)

        out: Dict[str, Any] = {"type": "questions", "questions": [], "questions_text": ""}
        prev_stage = state.stage
        for _ in range(max_stage_advances):
            out = self.design_agent.step(state)
            if out["type"] == "report":
                break
            if state.stage == prev_stage:
                # This call did not advance the stage -> its real
                # required_fields are genuinely unmet. Stop here; whatever
                # DesignAgent just returned is the real blocker.
                break
            prev_stage = state.stage
            # else: stage advanced (possibly carrying incidental
            # non-blocking gaps) -- keep driving without new user input.

        return {"state": state, "result": out}

    # ------------------------------------------------------------------
    # ProblemSpec -> ProblemStatement (see module docstring for the table)
    # ------------------------------------------------------------------

    def to_problem_statement(
        self,
        spec: ProblemSpec,
        *,
        kind: Optional[str] = None,
        observations: Any = None,
        **overrides: Any,
    ) -> ProblemStatement:
        """Real, field-by-field mapping from a completed ``ProblemSpec`` to
        a real, valid ``ProblemStatement``. See the module docstring for the
        full table and reasoning; this docstring covers the call contract.

        Parameters
        ----------
        spec : ProblemSpec
            A completed (or at least intake-complete) elicited spec.
        kind : str or None
            Force the orchestrator ``kind`` (one of
            ``"forward"``/``"inverse"``/``"design"``/``"forecast"``/
            ``"uncertainty"``/``"discovery"``/``"digital_twin"``/
            ``"world_model"``). Required when ``spec.task_type`` has no
            confident mapping (see ``_TASK_TYPE_TO_KIND``); optional
            otherwise (overrides the table).
        observations : Any
            Forwarded to ``ProblemStatement.observations``. Required when
            the resolved ``kind`` is ``"inverse"`` (see module docstring).
        **overrides
            Any other ``ProblemStatement`` field, forwarded verbatim and
            taking precedence over every default/derived value computed
            here (e.g. ``scenarios=[...]``, ``pde_hint="heat_2d"``,
            ``n_samples=500``, ``device="cuda"``, ``extra={...}``).

        Raises
        ------
        ValueError
            If ``spec.task_type`` has no confident mapping and ``kind`` was
            not supplied, or if the resolved ``kind`` is ``"inverse"`` and
            no ``observations`` were supplied -- both are the
            non-invention-policy-style "raise a clear, actionable error"
            path rather than guessing.
        """
        resolved_kind = kind or _TASK_TYPE_TO_KIND.get(spec.task_type)
        if resolved_kind is None:
            raise ValueError(
                f"ProblemSpec.task_type={spec.task_type!r} has no confident, "
                "non-invented mapping to a PhysicsOrchestrator ProblemStatement "
                f"kind (only {sorted(_TASK_TYPE_TO_KIND)} map unambiguously). "
                f"Pass kind=... explicitly (one of {_ORCHESTRATOR_KINDS}) to "
                "to_problem_statement()/run() to choose one."
            )
        if resolved_kind not in _ORCHESTRATOR_KINDS:
            raise ValueError(
                f"kind={resolved_kind!r} is not a PhysicsOrchestrator kind. "
                f"Valid kinds: {_ORCHESTRATOR_KINDS}."
            )

        description = (spec.goal or "").strip() or (spec.title or "").strip()

        kwargs: Dict[str, Any] = dict(kind=resolved_kind, description=description)

        if "domain_hint" not in overrides:
            raw_domain = (spec.geometry.domain or "").strip()
            # domain_hint is verified (full read of orchestrator.py) to be
            # inert metadata -- no _plan_* method consumes it -- so this
            # lossy slug can never silently change which tools execute.
            kwargs["domain_hint"] = (
                raw_domain.lower().replace(" ", "_") if raw_domain else "unit_square"
            )

        if observations is not None:
            kwargs["observations"] = observations
        elif resolved_kind == "inverse" and "observations" not in overrides:
            raise ValueError(
                "kind='inverse' requires observation data for "
                "PhysicsOrchestrator._plan_inverse (it raises ValueError "
                "without it), but ProblemSpec carries no actual observation "
                "payload -- only descriptive metadata in ProblemSpec.data. "
                "Pass observations=... explicitly to "
                "to_problem_statement()/run()."
            )

        # pde_hint / scenarios: deliberately NOT derived -- see module
        # docstring ("(none) pde_hint, scenarios" row) for why bridging
        # ProblemSpec/PINN-pde-kind vocabulary to pinneapple_worldmodel's
        # scenario-name vocabulary would require guessing. Left unset here
        # so ProblemStatement's own defaults apply unless overridden.

        if "output" not in overrides:
            output = ["model"]
            if spec.validation.primary_metrics or spec.validation.acceptance_criteria:
                output.append("validation_report")
            kwargs["output"] = output

        # params, model, n_samples, n_steps, device, save_dir, verbose,
        # extra: no natural source field in today's ProblemSpec (see module
        # docstring) -- left to ProblemStatement's own defaults unless
        # present in **overrides below.

        kwargs.update(overrides)
        return ProblemStatement(**kwargs)

    # ------------------------------------------------------------------
    # End-to-end: plain English -> real executed physics result
    # ------------------------------------------------------------------

    def run(self, user_text: str, **elicitation_kwargs: Any) -> UnifiedAgentResult:
        """Elicit from *user_text*, then actually execute against a real
        ``PhysicsOrchestrator`` if (and only if) elicitation completed.

        Parameters
        ----------
        user_text : str
            A single starting problem description. If it is detailed enough
            for ``DesignAgent`` to fill every ``required_fields`` blocker
            across every elicitation stage from this one message (see
            ``elicitation/stages.py``), this returns
            ``status == "executed"``. Otherwise it returns
            ``status == "needs_more_info"`` with the real gaps/questions
            ``DesignAgent`` surfaced -- this method never fabricates a
            multi-turn conversation's answers.
        **elicitation_kwargs
            ``kind`` : str or None -- forwarded to ``to_problem_statement``.
            ``observations`` : Any -- forwarded to ``to_problem_statement``.
            ``statement_overrides`` : dict -- extra ``ProblemStatement``
                keyword overrides, forwarded to ``to_problem_statement``.
            ``initial_spec`` : ProblemSpec or None -- forwarded to
                ``drive_elicitation`` (see its docstring: lets a caller
                supply nested ``ProblemSpec`` sub-objects that the live
                chat-extraction path cannot reach today).
            ``max_stage_advances`` : int or None -- forwarded to
                ``drive_elicitation``.

        Returns
        -------
        UnifiedAgentResult
        """
        kind = elicitation_kwargs.pop("kind", None)
        observations = elicitation_kwargs.pop("observations", None)
        statement_overrides = elicitation_kwargs.pop("statement_overrides", {}) or {}
        initial_spec = elicitation_kwargs.pop("initial_spec", None)
        max_stage_advances = elicitation_kwargs.pop("max_stage_advances", None)
        if elicitation_kwargs:
            raise TypeError(
                f"run() got unexpected keyword arguments: {sorted(elicitation_kwargs)}"
            )

        driven = self.drive_elicitation(
            user_text,
            initial_spec=initial_spec,
            max_stage_advances=max_stage_advances,
        )
        state: DesignState = driven["state"]
        out: Dict[str, Any] = driven["result"]

        if out["type"] != "report":
            return UnifiedAgentResult(
                status="needs_more_info",
                design_state=state,
                gaps=list(out.get("questions", state.unresolved_gaps())),
                questions_text=out.get("questions_text", ""),
            )

        report: DesignReport = out["report"]

        try:
            problem_statement = self.to_problem_statement(
                report.spec,
                kind=kind,
                observations=observations,
                **statement_overrides,
            )
        except Exception as exc:  # real error, surfaced not swallowed
            return UnifiedAgentResult(
                status="execution_failed",
                design_report=report,
                design_state=state,
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            execution_result = self.orchestrator.solve(problem_statement)
        except Exception as exc:  # real error, surfaced not swallowed
            return UnifiedAgentResult(
                status="execution_failed",
                design_report=report,
                problem_statement=problem_statement,
                design_state=state,
                error=f"{type(exc).__name__}: {exc}",
            )

        return UnifiedAgentResult(
            status="executed",
            design_report=report,
            problem_statement=problem_statement,
            execution_result=execution_result,
            design_state=state,
        )
