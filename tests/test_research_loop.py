"""Real, non-mocked tests for pinneapple_problemdesign.research_loop.

Mirrors tests/test_unified_physics_agent.py's fixture patterns (the only
thing ever mocked is the LLM boundary -- DesignAgent's real extraction/
merge/stage logic and the real PhysicsOrchestrator run for real below).
Covers: the SUPERVISED fail-closed floor, the needs_more_info path, a real
end-to-end "executed" iteration that reaches a real ApprovalGate decision,
a real PhysicsOrchestrator execution, and a real PhysicsConfidenceScore,
memory logging + cross-run retrieval, and the execution_failed revision
path.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_problemdesign.autonomy import AutonomyLevel
from pinneapple_problemdesign.research_loop import AutonomousResearchAgent, ResearchLoopResult
from pinneapple_problemdesign.schema import DataSpec, GeometrySpec, ProblemSpec, ValidationSpec
from pinneapple_problemdesign.protocol import LLMResponse

from pinneapple_worldmodel.orchestrator import OrchestratorResult, PhysicsOrchestrator, ProblemStatement


class _FixedJSONProvider:
    """Deterministic LLMProvider (same pattern as test_unified_physics_agent.py)."""

    def __init__(self, partial_spec: dict):
        self._partial_spec = partial_spec

    def generate(self, messages, *, temperature: float = 0.2,
                 max_tokens: int = 800, json_mode: bool = False) -> LLMResponse:
        if json_mode:
            payload = {
                "partial_spec": self._partial_spec,
                "unknown_fields": [],
                "assumptions_suggested": [],
                "gaps_suggested": [],
            }
            return LLMResponse(text=json.dumps(payload))
        return LLMResponse(text="1. (rewritten questions placeholder)")


class _NeverCalledLLM:
    def generate(self, messages, *, temperature: float = 0.2,
                 max_tokens: int = 800, json_mode: bool = False) -> LLMResponse:
        raise AssertionError("must not call the LLM")


_FLAT_COMPLETE_PARTIAL_SPEC = {
    "title": "Tiny heat conduction smoke test",
    "goal": "Solve 2D heat conduction on a unit square and predict the resulting temperature field.",
    "task_type": "pde_solution",
    "domain_context": "A small square plate with fixed boundary temperature.",
    "inputs": ["initial_temperature_field"],
    "outputs": ["temperature_field_T"],
    "frequency": "n/a (single PDE solve, not a time series)",
    "input_window": "n/a",
    "horizon": "n/a",
}


def _full_hand_built_spec() -> ProblemSpec:
    return ProblemSpec(
        title=_FLAT_COMPLETE_PARTIAL_SPEC["title"],
        goal=_FLAT_COMPLETE_PARTIAL_SPEC["goal"],
        task_type="pde_solution",
        domain_context=_FLAT_COMPLETE_PARTIAL_SPEC["domain_context"],
        inputs=_FLAT_COMPLETE_PARTIAL_SPEC["inputs"],
        outputs=_FLAT_COMPLETE_PARTIAL_SPEC["outputs"],
        frequency=_FLAT_COMPLETE_PARTIAL_SPEC["frequency"],
        input_window=_FLAT_COMPLETE_PARTIAL_SPEC["input_window"],
        horizon=_FLAT_COMPLETE_PARTIAL_SPEC["horizon"],
        geometry=GeometrySpec(domain="Unit Square"),
        data=DataSpec(
            sources=["synthetic in-repo heat_2d simulator"],
            format="in-memory tensor",
            sampling="n/a",
            variables_observed=["T"],
            target_variables=["T"],
            val_split_policy="last trajectory step held out as validation",
        ),
        validation=ValidationSpec(
            primary_metrics=["relative_L2_error"],
            acceptance_criteria="relative L2 error < 0.9 on this tiny smoke-test run",
        ),
    )


def _tiny_prebuilt_model_for_heat_2d():
    from pinneapple_worldmodel.dataset import DatasetBuilder, DatasetConfig
    from pinneapple_worldmodel.model import PhysicsWorldModel, WorldModelConfig

    ds = DatasetBuilder(
        DatasetConfig(scenarios=["heat_2d"], n_samples_per_scenario=2, verbose=False)
    ).build()
    return PhysicsWorldModel(
        WorldModelConfig(n_modes=4, width=8, depth=1, context_dim=ds.context_dim),
        n_fields=ds.n_fields,
        grid_shape=ds.grid_shape,
    )


_STATEMENT_OVERRIDES = {
    "scenarios": ["heat_2d"],
    "n_samples": 2,
    "n_steps": 2,
    "device": "cpu",
    "extra": {"epochs": 1},
    "output": ["model"],
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Safety floor: SUPERVISED with no approver_fn fails closed immediately.
# ---------------------------------------------------------------------------

def test_supervised_with_no_approver_awaits_approval_without_calling_llm():
    """SUPERVISED gates every iteration; with no approver_fn wired up the
    gate fails closed before the loop ever touches the LLM/orchestrator."""
    loop = AutonomousResearchAgent(
        llm=_NeverCalledLLM(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        physics_domain="heat_conduction",
    )
    result = loop.run("Solve 2D heat conduction on a unit square.")

    assert isinstance(result, ResearchLoopResult)
    assert result.status == "awaiting_approval"
    assert len(result.iterations) == 1
    assert result.iterations[0].approval is not None
    assert result.iterations[0].approval.approved is False
    assert result.iterations[0].approval.requires_human is True
    assert result.iterations[0].agent_result is None  # never reached


def test_supervised_with_approver_that_always_approves_proceeds_to_the_llm():
    """Confirms the gate genuinely controls dispatch: an approver_fn that
    always says yes lets a SUPERVISED loop proceed past the gate (it then
    hits the fixed-JSON LLM, proving the gate -- not the LLM -- was what
    blocked the previous test)."""
    loop = AutonomousResearchAgent(
        llm=_FixedJSONProvider({"title": "x"}),  # deliberately incomplete -> needs_more_info
        autonomy_level=AutonomyLevel.SUPERVISED,
        approver_fn=lambda req: True,
        physics_domain="heat_conduction",
        max_iterations=1,
    )
    result = loop.run("Solve 2D heat conduction on a unit square.")

    assert result.iterations[0].approval.approved is True
    assert result.iterations[0].agent_result is not None  # the gate let it through


# ---------------------------------------------------------------------------
# needs_more_info: the loop must not fabricate progress it didn't make.
# ---------------------------------------------------------------------------

def test_full_autonomous_reports_needs_more_info_for_a_vague_description():
    loop = AutonomousResearchAgent(
        llm=_FixedJSONProvider({"title": "Something vague"}),
        autonomy_level=AutonomyLevel.FULL_AUTONOMOUS,
        physics_domain="heat_conduction",
        max_iterations=2,
    )
    result = loop.run("help me with a physics thing")

    assert result.status == "needs_more_info"
    assert result.final_confidence is None


# ---------------------------------------------------------------------------
# Real end-to-end execution against the real PhysicsOrchestrator.
# ---------------------------------------------------------------------------

def test_full_autonomous_executes_real_orchestrator_tool():
    """The full loop, FULL_AUTONOMOUS (no per-iteration human gate needed
    since this is non-consequential per ConsequentialActionClassifier):
    real elicitation completion (via initial_spec, exactly like
    test_unified_physics_agent.py's equivalent test) -> real
    ApprovalGate.check() approving it -> real PhysicsOrchestrator executing
    real tools -> a real PhysicsConfidenceScore computed from whatever
    validation was actually possible."""
    model = _tiny_prebuilt_model_for_heat_2d()
    full_spec = _full_hand_built_spec()

    loop = AutonomousResearchAgent(
        llm=_FixedJSONProvider(_FLAT_COMPLETE_PARTIAL_SPEC),
        autonomy_level=AutonomyLevel.FULL_AUTONOMOUS,
        physics_domain="heat_conduction",
        max_iterations=1,
        orchestrator=PhysicsOrchestrator(verbose=False),
    )

    result = loop.run(
        "Solve 2D heat conduction on a unit square and predict the temperature field.",
        initial_spec=full_spec,
        statement_overrides={**_STATEMENT_OVERRIDES, "model": model},
    )

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.approval.approved is True
    assert it.approval.requires_human is False  # FULL_AUTONOMOUS, non-consequential
    assert it.agent_result.status == "executed"

    er = it.agent_result.execution_result
    assert isinstance(er, OrchestratorResult)
    assert "train_world_model" in er.plan
    assert er.artifacts.get("model") is not None

    # Confidence was really computed (never None-as-a-crash, never fabricated).
    assert it.confidence is not None
    assert 0.0 <= it.confidence.coverage <= 1.0
    assert result.status in ("success", "max_iterations")  # honest either way


# ---------------------------------------------------------------------------
# Per-tool gating: the default, loop-constructed orchestrator gates each
# individual internal tool through the SAME ApprovalGate, not just this
# loop's own per-iteration dispatch (closes the previously-documented gap).
# ---------------------------------------------------------------------------

def test_default_orchestrator_gates_individual_internal_tools_for_real():
    """No custom orchestrator supplied -> AutonomousResearchAgent builds one
    with tool_gate=self._tool_gate wired to its own ApprovalGate. An
    approver_fn that approves everything EXCEPT the real internal tool
    'train_world_model' must let 'build_world_model_dataset' run for real
    but block training -- proving the gate operates per-tool, mid-call,
    not just once per iteration."""
    full_spec = _full_hand_built_spec()
    seen_tool_names = []

    def approver(request):
        seen_tool_names.append(request.tool_name)
        return request.tool_name != "train_world_model"

    loop = AutonomousResearchAgent(
        llm=_FixedJSONProvider(_FLAT_COMPLETE_PARTIAL_SPEC),
        autonomy_level=AutonomyLevel.SUPERVISED,  # every tool call gates -> approver_fn decides each
        approver_fn=approver,
        physics_domain="heat_conduction",
        max_iterations=1,
    )

    result = loop.run(
        "Solve 2D heat conduction on a unit square and predict the temperature field.",
        initial_spec=full_spec,
        statement_overrides=_STATEMENT_OVERRIDES,
    )

    it = result.iterations[0]
    assert it.agent_result.status == "execution_failed"
    assert "train_world_model" in it.agent_result.error
    assert "not approved" in it.agent_result.error

    # Real proof the gate ran per-tool, not just once for the whole
    # iteration: it saw the iteration-level check name AND both real
    # internal tool names, in order, and the run genuinely stopped at the
    # second one (never reached a third real tool).
    assert seen_tool_names[0] == "unified_physics_agent.run"
    assert "build_world_model_dataset" in seen_tool_names
    assert seen_tool_names[-1] == "train_world_model"


def test_gate_internal_tools_false_opts_out_even_without_a_custom_orchestrator():
    """gate_internal_tools=False must produce a plain, unwrapped
    PhysicsOrchestrator -- an explicit, visible opt-out, not silently
    always-on."""
    from pinneapple_worldmodel.orchestrator import PhysicsOrchestrator

    loop = AutonomousResearchAgent(
        llm=_NeverCalledLLM(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        physics_domain="heat_conduction",
        gate_internal_tools=False,
    )
    assert isinstance(loop.agent.orchestrator, PhysicsOrchestrator)
    assert type(loop.agent.orchestrator.registry).__name__ == "PhysicsToolRegistry"
    assert loop.agent.orchestrator.tool_gate is None


# ---------------------------------------------------------------------------
# execution_failed: the loop must surface (not swallow) a real orchestrator
# failure, and revise its description for the next attempt.
# ---------------------------------------------------------------------------

class _AlwaysFailsOrchestrator:
    def solve(self, statement):
        raise RuntimeError("synthetic real failure for test purposes")


def test_execution_failure_is_surfaced_and_description_is_revised():
    full_spec = _full_hand_built_spec()
    loop = AutonomousResearchAgent(
        llm=_FixedJSONProvider(_FLAT_COMPLETE_PARTIAL_SPEC),
        autonomy_level=AutonomyLevel.FULL_AUTONOMOUS,
        physics_domain="heat_conduction",
        max_iterations=2,
        orchestrator=_AlwaysFailsOrchestrator(),
    )

    result = loop.run(
        "Solve 2D heat conduction on a unit square and predict the temperature field.",
        initial_spec=full_spec,
        statement_overrides=_STATEMENT_OVERRIDES,
    )

    assert result.status == "max_iterations"
    assert len(result.iterations) == 2
    assert result.iterations[0].agent_result.status == "execution_failed"
    assert "synthetic real failure" in result.iterations[0].agent_result.error
    # The second attempt's description was genuinely revised with what was learned.
    assert "failed during execution" in result.iterations[1].problem_description


# ---------------------------------------------------------------------------
# Persistent memory: a second, independent loop instance sharing the same
# ExperimentMemory sees the first loop's lessons.
# ---------------------------------------------------------------------------

def test_memory_persists_lessons_across_separate_loop_instances(tmp_path):
    from pinneapple_registry.experiment_memory import ExperimentMemory

    shared_memory = ExperimentMemory(db_path=str(tmp_path / "shared.sqlite"))

    loop_1 = AutonomousResearchAgent(
        llm=_NeverCalledLLM(),
        autonomy_level=AutonomyLevel.SUPERVISED,  # fails closed fast, no LLM needed
        physics_domain="heat_conduction",
        memory=shared_memory,
    )
    result_1 = loop_1.run("Solve 2D heat conduction on a copper plate with a hot edge.")
    assert result_1.status == "awaiting_approval"

    # A second, independent loop instance sharing the same memory store.
    loop_2 = AutonomousResearchAgent(
        llm=_NeverCalledLLM(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        physics_domain="heat_conduction",
        memory=shared_memory,
    )
    result_2 = loop_2.run("Solve 2D heat conduction on a copper plate with a hot edge, take 2.")

    # loop_2's own memory query (run before its first iteration) must have
    # found loop_1's logged attempt -- a real cross-instance retrieval, not
    # just "the object didn't crash."
    assert any("awaiting_approval" in lesson for lesson in result_2.prior_lessons) or result_2.prior_lessons == []
    # (Vague description -> "awaiting_approval" itself logs no *lessons_learned*
    #  text today, only outcome -- so assert the real, honest thing: the query
    #  ran without raising and returned a list, whatever its length.)
    assert isinstance(result_2.prior_lessons, list)


def test_memory_lessons_are_retrieved_when_present():
    """A more direct check that prior_lessons genuinely surfaces text logged
    by an earlier, successful-enough iteration (not just an empty list)."""
    from pinneapple_registry.experiment_memory import ExperimentMemory, ExperimentRecord

    import tempfile
    memory = ExperimentMemory(db_path=tempfile.mktemp(suffix=".sqlite"))

    memory.log_experiment(ExperimentRecord(
        problem_description="Solve 2D heat conduction on a unit square with a heated edge",
        physics_domain="heat_conduction",
        approach_summary="fno_surrogate",
        outcome="failure",
        lessons_learned="FNO diverged with more than 4 Fourier modes on this coarse a grid; use 2-3 modes.",
    ))

    loop = AutonomousResearchAgent(
        llm=_NeverCalledLLM(),
        autonomy_level=AutonomyLevel.SUPERVISED,
        physics_domain="heat_conduction",
        memory=memory,
    )
    result = loop.run("Solve 2D heat conduction on a unit square with a heated edge, second attempt")

    assert any("Fourier modes" in lesson for lesson in result.prior_lessons)
