"""``AutonomousResearchAgent``: the autonomous research cycle on top of
``UnifiedPhysicsAgent``.

This closes the loop the rest of this package's pieces were built for,
composing them exactly as designed -- none of their internals are touched:

* :class:`~.unified_agent.UnifiedPhysicsAgent` -- plain-English problem
  description -> real elicitation -> real ``PhysicsOrchestrator`` execution.
* :class:`~.autonomy.ApprovalGate` -- the non-negotiable safety floor.
  Every iteration's dispatch is gated through it before the loop is allowed
  to actually run the orchestrator. See the honest limitation note below.
* :class:`pinneapple_registry.experiment_memory.ExperimentMemory` --
  persistent cross-run memory: each attempt is logged, and prior similar
  attempts (successes AND failures) are consulted before a new one starts.
* :func:`pinneapple_analysis.verification.physics_confidence_score.compute_physics_confidence`
  and :class:`pinneapple_llm.guardrail.PhysicsGuardrail` -- real,
  already-executed-check-based validation of what the loop just produced,
  used to decide "good enough, stop" vs. "keep going."

The cycle, per iteration
-------------------------
::

    query memory for similar past attempts (iteration 0 only)
        -> gate this iteration's dispatch through ApprovalGate
        -> UnifiedPhysicsAgent.run(problem_description)
        -> if executed: run PhysicsGuardrail on the resulting model (if any)
                        -> compute_physics_confidence(...)
        -> log an ExperimentRecord (outcome, confidence, lessons)
        -> confidence >= confidence_target?  -> stop, status="success"
        -> max_iterations reached?           -> stop, status="max_iterations"
        -> otherwise: revise the problem description with what was just
           learned (a plain, inspectable text amendment -- never a silent
           spec mutation) and loop again

Honest limitation: gating granularity
---------------------------------------
:class:`~.autonomy.ApprovalGate` is designed to gate individual tool calls,
but ``PhysicsOrchestrator.solve()`` (unmodified, per this package's
composition-only design rule -- see ``unified_agent``'s module docstring)
has no hook to gate each *internal* tool it selects and runs. This loop
therefore gates at the coarser granularity of "may this iteration dispatch
to the orchestrator at all" -- once approved, whatever real tools
``PhysicsOrchestrator`` internally chooses for that call run without a
further per-tool gate from *this* loop. This is a real, load-bearing
limitation, not swept under the rug: it means a consequential tool that a
future ``PhysicsToolRegistry`` addition causes the orchestrator to select
*internally* is not separately re-gated mid-call by this loop. Closing that
gap fully would require adding a gating hook inside
``PhysicsOrchestrator`` itself -- a change to that live system, deliberately
out of scope for this composition-only module. Until that exists, treat
``FULL_AUTONOMOUS``/``SEMI_AUTONOMOUS`` runs of this loop as trusting
``PhysicsToolRegistry``'s current, real tool set (all local
simulation/compute, per ``autonomy.py``'s own grounding) -- not as a
guarantee against a not-yet-registered consequential tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .autonomy import ApprovalDecision, ApprovalGate, AutonomyLevel
from .protocol import LLMProvider
from .unified_agent import UnifiedAgentResult, UnifiedPhysicsAgent

try:
    from pinneapple_registry.experiment_memory import ExperimentMemory, ExperimentRecord
    _HAS_MEMORY = True
except Exception:  # pragma: no cover - pinneapple_registry is a real, but optional-at-import-time, sibling package
    _HAS_MEMORY = False

try:
    from pinneapple_llm.guardrail import PhysicsGuardrail, GuardrailReport
    _HAS_GUARDRAIL = True
except Exception:  # pragma: no cover
    _HAS_GUARDRAIL = False

from pinneapple_analysis.verification.physics_confidence_score import (
    PhysicsConfidenceScore,
    compute_physics_confidence,
)


@dataclass
class IterationRecord:
    """What happened during one iteration of the loop -- kept in full so a
    human reviewing a run afterwards can see every decision, not just the
    final outcome."""

    iteration: int
    problem_description: str
    approval: Optional[ApprovalDecision] = None
    agent_result: Optional[UnifiedAgentResult] = None
    guardrail_report: "Optional[GuardrailReport]" = None
    confidence: Optional[PhysicsConfidenceScore] = None
    note: str = ""


@dataclass
class ResearchLoopResult:
    """The full, inspectable outcome of an :meth:`AutonomousResearchAgent.run` call."""

    status: str
    """One of ``"success"``, ``"max_iterations"``, ``"needs_more_info"``,
    ``"awaiting_approval"``, ``"execution_failed"``."""
    iterations: List[IterationRecord] = field(default_factory=list)
    final_confidence: Optional[PhysicsConfidenceScore] = None
    final_result: Optional[UnifiedAgentResult] = None
    memory_record_ids: List[str] = field(default_factory=list)
    prior_lessons: List[str] = field(default_factory=list)


class AutonomousResearchAgent:
    """The Phase B research loop: hypothesize (revise description) ->
    simulate (``UnifiedPhysicsAgent``) -> validate (``PhysicsGuardrail`` /
    ``PhysicsConfidenceScore``) -> decide (stop or revise) -> repeat, behind
    ``ApprovalGate``'s non-negotiable safety floor, with
    ``ExperimentMemory`` learning across runs.

    Every real decision point in this class delegates to a real, separately
    built and tested module -- this class only sequences them; it never
    reimplements elicitation, execution, safety classification, similarity
    search, or physics validation.

    Parameters
    ----------
    llm : LLMProvider
        Passed through to the internal ``UnifiedPhysicsAgent``.
    autonomy_level : AutonomyLevel
        Passed through to the internal ``ApprovalGate``.
    approver_fn : callable or None
        Passed through to the internal ``ApprovalGate``. If ``None``, any
        action requiring human approval is refused (fail closed) -- see
        ``ApprovalGate``'s own docstring. This includes
        ``SUPERVISED``-level runs, which gate *every* iteration: a
        ``SUPERVISED`` loop with no ``approver_fn`` will refuse to run past
        iteration 1 unless the caller supplies one.
    memory : ExperimentMemory or None
        If ``None`` and ``pinneapple_registry`` is importable, a real
        ``ExperimentMemory`` backed by a fresh temp-file store is
        constructed lazily. Pass an explicit instance to share memory
        across multiple ``AutonomousResearchAgent`` runs/sessions. If
        ``pinneapple_registry`` genuinely isn't importable, memory is
        skipped entirely (the loop still runs -- memory is a real
        enhancement, not a hard dependency of the core cycle).
    physics_domain : str
        Free-text tag used for memory logging/retrieval (e.g.
        ``"heat_conduction"``, ``"cfd"``) -- never inferred/guessed; the
        caller supplies it.
    max_iterations : int
        Hard cap on how many times the loop will revise and retry.
    confidence_target : float
        Stop with ``status="success"`` once
        ``PhysicsConfidenceScore.overall_score`` is not ``None`` and
        ``>= confidence_target``. Note ``overall_score`` can be ``None``
        (zero validation components available) -- such a run never meets
        the target on confidence alone and will run to ``max_iterations``,
        which is itself an honest signal ("this loop could never actually
        validate what it produced"), not a bug.
    **approval_gate_kwargs
        Forwarded to ``ApprovalGate`` (``iteration_report_every``,
        ``confidence_pause_threshold``, ``classifier``).
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
        approver_fn=None,
        memory: "Optional[ExperimentMemory]" = None,
        physics_domain: str = "general",
        max_iterations: int = 5,
        confidence_target: float = 0.7,
        orchestrator=None,
        **approval_gate_kwargs: Any,
    ) -> None:
        self.agent = UnifiedPhysicsAgent(llm, orchestrator=orchestrator)
        self.gate = ApprovalGate(autonomy_level, approver_fn=approver_fn, **approval_gate_kwargs)
        self.physics_domain = physics_domain
        self.max_iterations = max_iterations
        self.confidence_target = confidence_target

        self._memory: "Optional[ExperimentMemory]" = memory
        if self._memory is None and memory is None and _HAS_MEMORY:
            self._memory = None  # constructed lazily in `memory` property

    @property
    def memory(self) -> "Optional[ExperimentMemory]":
        """The real ``ExperimentMemory`` this loop logs to/queries, or
        ``None`` if ``pinneapple_registry`` isn't importable. Constructed
        lazily (a fresh temp-file-backed store) on first access if the
        caller didn't supply one."""
        if self._memory is None and _HAS_MEMORY:
            import tempfile

            self._memory = ExperimentMemory(
                db_path=tempfile.mktemp(prefix="pinneapple_research_loop_", suffix=".sqlite")
            )
        return self._memory

    # ------------------------------------------------------------------

    def run(self, problem_description: str, **elicitation_kwargs: Any) -> ResearchLoopResult:
        """Run the full hypothesize -> simulate -> validate -> revise cycle.

        ``elicitation_kwargs`` are forwarded to ``UnifiedPhysicsAgent.run``
        on every iteration (``kind``, ``observations``,
        ``statement_overrides``, ``initial_spec``, ``max_stage_advances``
        -- see that method's docstring).
        """
        result = ResearchLoopResult(status="max_iterations")

        prior_lessons: List[str] = []
        if self.memory is not None:
            try:
                similar = self.memory.query_similar(
                    problem_description, physics_domain=self.physics_domain, top_k=3
                )
                prior_lessons = [
                    rec.lessons_learned
                    for rec, _score in similar
                    if rec.lessons_learned
                ]
            except Exception:
                # Memory is a real enhancement, not load-bearing for the
                # core cycle -- a query failure (e.g. empty/corrupt store)
                # must not abort the whole loop.
                prior_lessons = []
        result.prior_lessons = prior_lessons

        current_description = problem_description
        if prior_lessons:
            lessons_text = "; ".join(prior_lessons)
            current_description = (
                f"{problem_description}\n\n"
                f"[From {len(prior_lessons)} similar prior attempt(s) in memory: {lessons_text}]"
            )

        prev_confidence: Optional[float] = None

        for i in range(1, self.max_iterations + 1):
            record = IterationRecord(iteration=i, problem_description=current_description)
            result.iterations.append(record)

            decision = self.gate.check(
                "unified_physics_agent.run",
                {"problem_description": current_description, "domain": self.physics_domain},
                current_iteration=i,
                current_confidence=prev_confidence,
            )
            record.approval = decision
            if not decision.approved:
                result.status = "awaiting_approval"
                self._log(record, current_description, outcome="awaiting_approval")
                return result

            agent_result = self.agent.run(current_description, **elicitation_kwargs)
            record.agent_result = agent_result
            result.final_result = agent_result

            if agent_result.status == "needs_more_info":
                result.status = "needs_more_info"
                self._log(record, current_description, outcome="needs_more_info")
                return result

            if agent_result.status == "execution_failed":
                record.note = f"execution failed: {agent_result.error}"
                self._log(
                    record,
                    current_description,
                    outcome="failure",
                    lessons=f"Execution failed: {agent_result.error}",
                )
                current_description = (
                    f"{problem_description}\n\n"
                    f"[Prior attempt {i} failed during execution: {agent_result.error}. "
                    "Consider a different approach.]"
                )
                prev_confidence = 0.0
                continue

            # status == "executed"
            confidence = self._validate(agent_result)
            record.confidence = confidence
            result.final_confidence = confidence

            score = confidence.overall_score
            prev_confidence = score if score is not None else 0.0

            outcome = "success" if (score is not None and score >= self.confidence_target) else "partial"
            lessons = (
                f"Iteration {i}: confidence={score}, coverage={confidence.coverage}, "
                f"components={[c.name for c in confidence.components]}."
            )
            self._log(record, current_description, outcome=outcome, lessons=lessons, confidence=score)

            if outcome == "success":
                result.status = "success"
                return result

            current_description = (
                f"{problem_description}\n\n"
                f"[Prior attempt {i} executed but confidence was "
                f"{score if score is not None else 'unavailable'} "
                f"(target {self.confidence_target}, coverage {confidence.coverage}). "
                "Try a different approach or gather more validation data.]"
            )

        return result

    # ------------------------------------------------------------------

    def _validate(self, agent_result: UnifiedAgentResult) -> PhysicsConfidenceScore:
        """Run whichever real validation checks are actually possible
        against ``agent_result.execution_result`` and aggregate them via
        ``compute_physics_confidence``. Never fabricates a check that
        didn't run -- if no model artifact is available, or
        ``PhysicsGuardrail`` isn't importable, or checking it raises, the
        guardrail component is simply omitted (not faked)."""
        guardrail_report = None
        if _HAS_GUARDRAIL and agent_result.execution_result is not None:
            artifacts: Dict[str, Any] = getattr(agent_result.execution_result, "artifacts", {}) or {}
            model = artifacts.get("model")
            if model is not None:
                try:
                    guardrail_report = PhysicsGuardrail().check(model)
                except Exception:
                    # A model artifact that PhysicsGuardrail cannot check
                    # (wrong shape/kind for its heuristics) is a real,
                    # non-fatal outcome -- the confidence score below will
                    # honestly reflect the resulting lower coverage rather
                    # than this loop crashing on it.
                    guardrail_report = None

        return compute_physics_confidence(guardrail_report=guardrail_report)

    def _log(
        self,
        record: IterationRecord,
        problem_description: str,
        *,
        outcome: str,
        lessons: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        if self.memory is None or not _HAS_MEMORY:
            return
        try:
            approach = "unified_physics_agent"
            if record.agent_result is not None and record.agent_result.problem_statement is not None:
                approach = f"unified_physics_agent(kind={record.agent_result.problem_statement.kind})"
            rec = ExperimentRecord(
                problem_description=problem_description,
                physics_domain=self.physics_domain,
                approach_summary=approach,
                outcome=outcome,
                confidence_score=confidence,
                metrics={},
                lessons_learned=lessons,
            )
            self.memory.log_experiment(rec)
        except Exception:
            # Logging failure must never abort a research iteration that
            # otherwise completed real work.
            pass
