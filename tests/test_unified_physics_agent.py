"""Real, non-mocked tests for pinneapple_problemdesign.unified_agent.

Covers:
  (a) to_problem_statement's field-by-field ProblemSpec -> ProblemStatement
      mapping, tested directly against hand-built ProblemSpecs (no LLM, no
      elicitation, no orchestrator involved) -- including the confident
      task_type->kind table, the domain_hint slugify, the validation_report
      output rule, and the two documented "raise a clear error" paths
      (unmapped task_type, inverse without observations).
  (b) UnifiedPhysicsAgent.run()'s "needs_more_info" path, for both a vague
      starting description and (honestly) a fully-detailed one -- the
      latter demonstrates a real, verified limitation of the live,
      unmodified pinneapple_problemdesign.merge.merge_into_spec: it can
      never populate ProblemSpec's nested dataclass sub-objects (DataSpec/
      ValidationSpec/...) from chat-extracted JSON, so no natural-language
      -only conversation can pass the "data"/"validation" elicitation
      stages today. Real gaps are asserted, not a crash.
  (c) UnifiedPhysicsAgent.run()'s "executed" path: a real end-to-end run
      that reaches a real DesignReport, builds a real ProblemStatement, and
      has the real, unmodified PhysicsOrchestrator actually execute real
      tools (build_world_model_dataset, train_world_model) from its real
      PhysicsToolRegistry -- nothing mocked below the LLM boundary.

None of pinneapple_problemdesign/agent.py, pinneapple_worldmodel/orchestrator.py,
or pinneapple_worldmodel/physics_tools.py are touched -- only exercised as-is.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_problemdesign import UnifiedPhysicsAgent, UnifiedAgentResult
from pinneapple_problemdesign.schema import (
    ProblemSpec,
    DataSpec,
    GeometrySpec,
    ValidationSpec,
    DesignReport,
    Gap,
)
from pinneapple_problemdesign.protocol import LLMResponse

from pinneapple_worldmodel.orchestrator import ProblemStatement, OrchestratorResult


# ---------------------------------------------------------------------------
# Fake LLM providers (the only thing mocked -- everything downstream of the
# LLM boundary, i.e. DesignAgent's real extraction/merge/stage logic and the
# real PhysicsOrchestrator, runs for real).
# ---------------------------------------------------------------------------

class _FixedJSONProvider:
    """Deterministic LLMProvider: always extracts the same fixed
    ``partial_spec`` regardless of the actual user text. Good enough to
    drive DesignAgent's real, unmodified extraction/merge pipeline through
    its real flat-field path without depending on an LLM's judgement calls
    (mirrors the pattern already used in
    examples/problem_designer/04_custom_provider_mock_end_to_end.py)."""

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
        # Question-rewriting step: a stable, non-empty placeholder is enough.
        return LLMResponse(text="1. (rewritten questions placeholder)")


class _NeverCalledLLM:
    """Proves to_problem_statement() never touches the LLM: any call fails
    the test loudly instead of silently returning nonsense."""

    def generate(self, messages, *, temperature: float = 0.2,
                 max_tokens: int = 800, json_mode: bool = False) -> LLMResponse:
        raise AssertionError("to_problem_statement() must not call the LLM")


# A fixed partial_spec covering every FLAT (non-nested) required field
# across the intake and io_and_time elicitation stages -- these two stages
# genuinely can be satisfied via DesignAgent's real chat-extraction path
# (see merge.py's _deep_merge: top-level flat fields merge correctly).
_FLAT_COMPLETE_PARTIAL_SPEC = {
    "title": "Tiny heat conduction smoke test",
    "goal": "Solve 2D heat conduction on a unit square and predict the resulting temperature field.",
    "task_type": "pde_solution",
    "domain_context": "A small square plate with fixed boundary temperature; used to smoke-test the full elicit-to-execute pipeline end to end.",
    "inputs": ["initial_temperature_field"],
    "outputs": ["temperature_field_T"],
    "frequency": "n/a (single PDE solve, not a time series)",
    "input_window": "n/a",
    "horizon": "n/a",
}


# ---------------------------------------------------------------------------
# (a) to_problem_statement: direct, focused unit-level mapping checks
# ---------------------------------------------------------------------------

def _complete_pde_spec() -> ProblemSpec:
    return ProblemSpec(
        title="Heat conduction",
        goal="Solve 2D heat conduction in a square domain.",
        task_type="pde_solution",
        geometry=GeometrySpec(domain="Unit Square"),
        validation=ValidationSpec(
            primary_metrics=["relative_L2_error"],
            acceptance_criteria="relative L2 error < 1e-2",
        ),
    )


def test_to_problem_statement_maps_pde_solution_field_by_field():
    """task_type='pde_solution' -> kind='forward'; goal -> description;
    geometry.domain -> slugified domain_hint; validation intent ->
    output includes both the baseline 'model' and 'validation_report'."""
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = _complete_pde_spec()

    ps = agent.to_problem_statement(spec)

    assert isinstance(ps, ProblemStatement)
    assert ps.kind == "forward"
    assert ps.description == spec.goal
    assert ps.domain_hint == "unit_square"
    assert "model" in ps.output
    assert "validation_report" in ps.output
    # No natural source for these -- must be ProblemStatement's own
    # untouched defaults (never invented).
    assert ps.pde_hint is None
    assert ps.scenarios == ProblemStatement().scenarios
    assert ps.params == {}
    assert ps.device == "cpu"


def test_to_problem_statement_description_falls_back_to_title():
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = ProblemSpec(title="Fallback Title", goal="", task_type="pde_solution")
    ps = agent.to_problem_statement(spec)
    assert ps.description == "Fallback Title"


def test_to_problem_statement_domain_hint_defaults_when_unset():
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = ProblemSpec(title="T", goal="G", task_type="pde_solution")
    ps = agent.to_problem_statement(spec)
    assert ps.domain_hint == "unit_square"


@pytest.mark.parametrize(
    "task_type,expected_kind",
    [
        ("forecasting", "forecast"),
        ("pde_solution", "forward"),
        ("optimization", "design"),
    ],
)
def test_to_problem_statement_confident_task_type_mapping_table(task_type, expected_kind):
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = ProblemSpec(title="T", goal="G", task_type=task_type)
    ps = agent.to_problem_statement(spec)
    assert ps.kind == expected_kind


def test_to_problem_statement_unmapped_task_type_raises_actionable_error():
    """task_type values with no unambiguous orchestrator kind (e.g.
    'control') must raise a clear error naming the gap, not guess."""
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = ProblemSpec(title="T", goal="G", task_type="control")

    with pytest.raises(ValueError, match="control"):
        agent.to_problem_statement(spec)

    # ...but an explicit override always works (caller's own judgement call).
    ps = agent.to_problem_statement(spec, kind="digital_twin")
    assert ps.kind == "digital_twin"


def test_to_problem_statement_inverse_without_observations_raises():
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = ProblemSpec(title="T", goal="G", task_type="inverse_problem")

    with pytest.raises(ValueError, match="observations"):
        agent.to_problem_statement(spec)

    sentinel_observations = object()
    ps = agent.to_problem_statement(spec, observations=sentinel_observations)
    assert ps.kind == "inverse"
    assert ps.observations is sentinel_observations


def test_to_problem_statement_overrides_take_precedence():
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())
    spec = _complete_pde_spec()
    ps = agent.to_problem_statement(
        spec, scenarios=["heat_2d"], device="cuda", n_samples=7, output=["model"],
    )
    assert ps.scenarios == ["heat_2d"]
    assert ps.device == "cuda"
    assert ps.n_samples == 7
    assert ps.output == ["model"]  # override wins over the validation_report rule


# ---------------------------------------------------------------------------
# (b) run(): needs_more_info path (vague, and honestly-detailed-but-still-
#     stuck-on-nested-fields) -- real gaps, never a crash.
# ---------------------------------------------------------------------------

def test_run_needs_more_info_for_vague_description():
    agent = UnifiedPhysicsAgent(llm=_FixedJSONProvider({}))

    result = agent.run("I want to predict something eventually.")

    assert isinstance(result, UnifiedAgentResult)
    assert result.status == "needs_more_info"
    assert result.design_report is None
    assert result.problem_statement is None
    assert result.execution_result is None
    assert result.gaps, "expected real unresolved Gap objects"
    assert all(isinstance(g, Gap) for g in result.gaps)
    assert result.questions_text
    # Genuinely stuck at the very first stage -- nothing was extracted.
    assert result.design_state.stage == "intake"


def test_run_needs_more_info_even_for_a_fully_detailed_description():
    """Honest limitation check: even a message from which every FLAT field
    extracts cleanly cannot finish elicitation through run() alone, because
    the live merge_into_spec cannot populate the nested `data`/`validation`
    ProblemSpec sub-objects from chat JSON (see unified_agent.py's
    drive_elicitation docstring). This proves (1) the composition correctly
    cascades through intake and io_and_time on real extracted data, and
    (2) it honestly reports needs_more_info rather than fabricating
    completion, exactly as the task requires."""
    agent = UnifiedPhysicsAgent(llm=_FixedJSONProvider(_FLAT_COMPLETE_PARTIAL_SPEC))

    result = agent.run(
        "Solve 2D heat conduction on a unit square and predict the temperature field. "
        "Sampling n/a, horizon n/a, input window n/a. Inputs: initial temperature field. "
        "Output: temperature field."
    )

    assert result.status == "needs_more_info"
    assert result.execution_result is None
    # Real progress: intake and io_and_time were satisfied by extraction,
    # so the state is genuinely stuck at "data" -- not stuck at "intake".
    assert result.design_state.stage == "data"
    assert any(g.id.startswith("data.") for g in result.gaps)


# ---------------------------------------------------------------------------
# (c) run(): the real, executed end-to-end path
# ---------------------------------------------------------------------------

def _tiny_prebuilt_model_for_heat_2d():
    """Build a real PhysicsWorldModel whose context_dim matches what
    DatasetBuilder will actually produce for scenarios=["heat_2d"],
    n_samples_per_scenario=2 (context_dim = len(param_keys) + len(pde_kinds)
    = 1 ("alpha") + 1 ("heat") = 2). PhysicsOrchestrator._plan_forward's
    real train_world_model tool only builds a *default* WorldModelConfig
    (context_dim=8) when no model is supplied, which would mismatch a
    single-scenario dataset's real context_dim -- passing a pre-built,
    correctly-sized model through ProblemStatement.model (a real, documented
    field: "pre-trained model to use as starting point") is the honest way
    to drive a fast, real, non-mocked orchestrator execution."""
    from pinneapple_worldmodel.dataset import DatasetBuilder, DatasetConfig
    from pinneapple_worldmodel.model import PhysicsWorldModel, WorldModelConfig

    ds = DatasetBuilder(
        DatasetConfig(scenarios=["heat_2d"], n_samples_per_scenario=2, verbose=False)
    ).build()
    model = PhysicsWorldModel(
        WorldModelConfig(n_modes=4, width=8, depth=1, context_dim=ds.context_dim),
        n_fields=ds.n_fields,
        grid_shape=ds.grid_shape,
    )
    return model


def _full_hand_built_spec() -> ProblemSpec:
    """A completed ProblemSpec covering every required_fields entry across
    every elicitation stage (intake/io_and_time/data/validation) -- the
    nested data/validation sub-objects are supplied directly here (see
    drive_elicitation's initial_spec docstring for why chat extraction
    alone cannot reach them)."""
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


def test_run_executes_real_orchestrator_tool_end_to_end():
    """The full loop: plain-English text (+ the structured nested data chat
    extraction cannot reach, via initial_spec) -> real DesignReport -> real
    ProblemStatement -> a real, unmodified PhysicsOrchestrator actually
    executing real tools from its real PhysicsToolRegistry."""
    model = _tiny_prebuilt_model_for_heat_2d()
    full_spec = _full_hand_built_spec()

    agent = UnifiedPhysicsAgent(
        llm=_FixedJSONProvider(_FLAT_COMPLETE_PARTIAL_SPEC),
        orchestrator_verbose=False,
    )

    result = agent.run(
        "Solve 2D heat conduction on a unit square and predict the temperature field.",
        initial_spec=full_spec,
        statement_overrides={
            "scenarios": ["heat_2d"],
            "model": model,
            "n_samples": 2,
            "n_steps": 2,
            "device": "cpu",
            "extra": {"epochs": 1},
            "output": ["model"],
            "verbose": False,
        },
    )

    assert result.status == "executed", (result.error, result.questions_text)
    assert result.error is None

    assert isinstance(result.design_report, DesignReport)
    assert result.design_state.done is True

    assert isinstance(result.problem_statement, ProblemStatement)
    assert result.problem_statement.kind == "forward"

    er = result.execution_result
    assert isinstance(er, OrchestratorResult)
    assert er.kind == "forward"
    # Real tools actually executed by the real PhysicsOrchestrator/
    # PhysicsToolRegistry -- not fabricated.
    assert "build_world_model_dataset" in er.plan
    assert "train_world_model" in er.plan
    assert "model" in er.artifacts
    assert "dataset" in er.artifacts
    assert er.artifacts["model"] is not None


def test_to_problem_statement_standalone_matches_what_run_used():
    """Focused unit check: to_problem_statement(report.spec) called directly
    (independent of run()'s orchestration) produces an equivalent, real,
    executable ProblemStatement for the same completed spec."""
    full_spec = _full_hand_built_spec()
    agent = UnifiedPhysicsAgent(llm=_NeverCalledLLM())

    ps = agent.to_problem_statement(full_spec, scenarios=["heat_2d"], device="cpu")

    assert ps.kind == "forward"
    assert ps.description == full_spec.goal
    assert ps.domain_hint == "unit_square"
    assert "validation_report" in ps.output
    assert ps.scenarios == ["heat_2d"]
