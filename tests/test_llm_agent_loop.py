"""Tests for ``pinneapple_llm.agent_loop`` -- a genuine multi-step LLM
agent loop where the LLM itself decides, turn by turn, which real tool
from a real :class:`~pinneapple_worldmodel.physics_tools.PhysicsToolRegistry`
to call next, observes the real result, and decides the next step.

Two tiers, mirroring ``tests/test_llm_cad_generation.py``'s convention:

1. Deterministic tests using a fake, dependency-injected ``llm_call_fn``
   (never a monkeypatched global) -- exercise prompt-building, live
   tool-metadata surfacing, JSON-response parsing/repair, real tool
   invocation via the registry, and error handling. These always run,
   no network/LLM backend required.
2. A real end-to-end test against a real local Ollama model, run only if
   one is actually reachable (skipped otherwise) -- asserts the loop
   actually invokes at least one real tool and produces a final answer.
"""
from __future__ import annotations

import json

import pytest

import pinneapple_llm as pl
from pinneapple_llm.agent_loop import (
    AgentLoopResult,
    AgentStep,
    _build_prompt,
    _describe_available_tools,
    _extract_first_json_object,
    _parse_agent_response,
    _strip_code_fences,
)


# ---------------------------------------------------------------------------
# A tiny fake tool registry -- exercises "any object with the same query
# interface", not just PhysicsToolRegistry itself.
# ---------------------------------------------------------------------------

class _FakeTool:
    def __init__(self, name, category, description, input_schema, fn):
        self.name = name
        self.category = category
        self.description = description
        self.input_schema = input_schema
        self._fn = fn

    def call(self, **kwargs):
        return self._fn(**kwargs)


class _FakeRegistry:
    """Minimal stand-in exposing the same ``available_tools()``/``get()``
    query interface as ``PhysicsToolRegistry``, with one tool that always
    succeeds and one that always raises -- so real-invocation success AND
    real-invocation failure are both exercised without touching the real
    (heavier) registry."""

    def __init__(self):
        self._tools = {
            "add_numbers": _FakeTool(
                "add_numbers", "math", "Add two numbers",
                {"a": "float", "b": "float"},
                lambda a, b: a + b,
            ),
            "always_fails": _FakeTool(
                "always_fails", "math", "A tool that always raises",
                {}, lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        }

    def available_tools(self):
        return list(self._tools.values())

    def get(self, name):
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools)}")
        return self._tools[name]


def _fake_llm_call(responses):
    """Returns an ``llm_call_fn(prompt, *, system)`` that yields each of
    *responses* in order across successive calls."""
    it = iter(responses)

    def _call(prompt, *, system=""):
        return next(it)

    return _call


# ---------------------------------------------------------------------------
# Parsing-helper unit tests
# ---------------------------------------------------------------------------

def test_strip_code_fences_removes_json_fence():
    text = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(text) == '{"a": 1}'


def test_extract_first_json_object_ignores_trailing_prose():
    text = 'Sure thing! {"action": "final_answer", "answer": "done"} Hope that helps.'
    assert _extract_first_json_object(text) == '{"action": "final_answer", "answer": "done"}'


def test_parse_agent_response_recovers_fenced_and_prefixed_json():
    raw = 'Here is my decision:\n```json\n{"action": "final_answer", "answer": "42"}\n```'
    parsed, err = _parse_agent_response(raw)
    assert err is None
    assert parsed == {"action": "final_answer", "answer": "42"}


def test_parse_agent_response_reports_error_on_garbage():
    parsed, err = _parse_agent_response("not json at all")
    assert parsed is None
    assert err is not None


def test_parse_agent_response_rejects_non_object_json():
    parsed, err = _parse_agent_response("[1, 2, 3]")
    assert parsed is None
    assert "JSON object" in err


# ---------------------------------------------------------------------------
# Live tool-metadata surfacing (real query interface, fake registry)
# ---------------------------------------------------------------------------

def test_describe_available_tools_pulls_live_metadata_not_hardcoded():
    described = _describe_available_tools(_FakeRegistry())
    names = {t["name"] for t in described}
    assert names == {"add_numbers", "always_fails"}
    add = next(t for t in described if t["name"] == "add_numbers")
    assert add["category"] == "math"
    assert add["expected_args"] == {"a": "float", "b": "float"}


def test_build_prompt_includes_goal_tools_and_history():
    tools = _describe_available_tools(_FakeRegistry())
    prompt = _build_prompt("Add 2 and 3", [], tools)
    assert "Add 2 and 3" in prompt
    assert "add_numbers" in prompt
    assert "no steps taken yet" in prompt


# ---------------------------------------------------------------------------
# run_agent_loop with an injected fake llm_call_fn -- deterministic,
# no network/LLM backend needed.
# ---------------------------------------------------------------------------

def test_agent_loop_calls_a_real_tool_then_final_answers():
    responses = [
        json.dumps({
            "action": "call_tool", "tool_name": "add_numbers",
            "tool_args": {"a": 2, "b": 3}, "reasoning": "need the sum first",
        }),
        json.dumps({
            "action": "final_answer", "answer": "5", "reasoning": "sum computed",
        }),
    ]
    result = pl.run_agent_loop(
        "Add 2 and 3",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_fake_llm_call(responses),
    )
    assert isinstance(result, AgentLoopResult)
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "5"
    assert len(result.steps) == 2

    tool_step = result.steps[0]
    assert isinstance(tool_step, AgentStep)
    assert tool_step.tool_name == "add_numbers"
    assert tool_step.tool_args == {"a": 2, "b": 3}
    assert tool_step.tool_result == 5
    assert tool_step.error is None
    assert tool_step.llm_reasoning == "need the sum first"

    final_step = result.steps[1]
    assert final_step.tool_name == "final_answer"
    assert final_step.tool_result == "5"


def test_agent_loop_records_real_tool_exception_without_crashing():
    responses = [
        json.dumps({"action": "call_tool", "tool_name": "always_fails", "tool_args": {}}),
        json.dumps({"action": "final_answer", "answer": "gave up after tool failed"}),
    ]
    result = pl.run_agent_loop(
        "Try the broken tool",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_fake_llm_call(responses),
    )
    assert result.stopped_reason == "final_answer"
    assert result.steps[0].tool_name == "always_fails"
    assert result.steps[0].tool_result is None
    assert "boom" in result.steps[0].error


def test_agent_loop_rejects_hallucinated_tool_name_without_crashing():
    responses = [
        json.dumps({"action": "call_tool", "tool_name": "not_a_real_tool", "tool_args": {}}),
        json.dumps({"action": "final_answer", "answer": "no such tool exists"}),
    ]
    result = pl.run_agent_loop(
        "Use a made-up tool",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_fake_llm_call(responses),
    )
    assert result.steps[0].tool_name == "not_a_real_tool"
    assert result.steps[0].error is not None
    assert "not found" in result.steps[0].error
    assert result.final_answer == "no such tool exists"


def test_agent_loop_recovers_from_malformed_json_and_keeps_going():
    responses = [
        "this is not valid JSON at all",
        json.dumps({"action": "final_answer", "answer": "recovered"}),
    ]
    result = pl.run_agent_loop(
        "Goal that trips up the model once",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_fake_llm_call(responses),
    )
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "recovered"
    assert len(result.steps) == 2
    assert result.steps[0].tool_name is None
    assert "Could not parse" in result.steps[0].error


def test_agent_loop_stops_at_max_steps_if_llm_never_finishes():
    always_call_tool = json.dumps({
        "action": "call_tool", "tool_name": "add_numbers", "tool_args": {"a": 1, "b": 1},
    })

    def _call(prompt, *, system=""):
        return always_call_tool

    result = pl.run_agent_loop(
        "Never-ending goal",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_call,
        max_steps=3,
    )
    assert result.stopped_reason == "max_steps"
    assert result.final_answer is None
    assert len(result.steps) == 3


def test_agent_loop_stops_unrecoverably_when_llm_call_itself_fails():
    def _call(prompt, *, system=""):
        raise ConnectionError("no LLM backend reachable")

    result = pl.run_agent_loop(
        "Goal with no LLM backend available",
        tool_registry=_FakeRegistry(),
        llm_call_fn=_call,
    )
    assert result.stopped_reason == "error"
    assert result.final_answer is None
    assert len(result.steps) == 1
    assert "LLM call failed" in result.steps[0].error


def test_agent_loop_defaults_to_a_real_physics_tool_registry():
    """No tool_registry passed -> a real, live PhysicsToolRegistry is
    constructed and its real tools are surfaced to the (fake) LLM."""
    seen_tool_names = {}

    def _call(prompt, *, system=""):
        # First call: inspect the prompt for real registered tool names,
        # then just declare done -- this test's point is that a REAL
        # registry backed the prompt, not that the LLM does anything.
        seen_tool_names["make_physics_domain" in prompt] = True
        return json.dumps({"action": "final_answer", "answer": "ok"})

    result = pl.run_agent_loop(
        "Inspect available tools",
        llm_call_fn=_call,
    )
    assert result.stopped_reason == "final_answer"
    assert seen_tool_names.get(True) is True


# ---------------------------------------------------------------------------
# Real local-model (Ollama) end-to-end test
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    try:
        import requests

        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_has_model(name: str) -> bool:
    try:
        import requests

        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        have = {m["name"] for m in r.json().get("models", [])}
        return name in have or f"{name}:latest" in have
    except Exception:
        return False


_OLLAMA_MODEL = "llama3.2:3b"
_skip_no_ollama = pytest.mark.skipif(
    not _ollama_reachable(), reason="no local Ollama server reachable at 127.0.0.1:11434"
)
_skip_no_model = pytest.mark.skipif(
    _ollama_reachable() and not _ollama_has_model(_OLLAMA_MODEL),
    reason=f"local Ollama server has no '{_OLLAMA_MODEL}' model pulled",
)


@_skip_no_ollama
@_skip_no_model
def test_local_llama_runs_a_real_agent_loop_and_invokes_a_real_tool():
    """End-to-end: a real local model, a real PhysicsToolRegistry, and a
    goal steered toward one trivially-callable real tool
    (``make_physics_domain``). Retries a few times, same reasoning as
    ``test_llm_cad_generation.py``'s real-model tests: a small local
    model is not 100% reliable at following the JSON action schema on
    every single attempt, but should get there within a few tries."""
    goal = (
        "Call the 'make_physics_domain' tool exactly once, with "
        "domain_type='unit_square', to create a physics domain. "
        "IMPORTANT: as soon as the HISTORY above shows that tool was "
        "already called (look for 'Step 1: called tool='make_physics_domain''), "
        "your very next response MUST use action 'final_answer' summarizing "
        "the result -- do NOT call any tool a second time."
    )

    last_result = None
    for _attempt in range(8):
        last_result = pl.run_agent_loop(
            goal, llm_provider="ollama", model=_OLLAMA_MODEL, max_steps=6,
        )
        invoked_real_tool = any(
            step.tool_name is not None
            and step.tool_name != "final_answer"
            and step.error is None
            for step in last_result.steps
        )
        if invoked_real_tool and last_result.stopped_reason == "final_answer":
            break

    assert last_result is not None
    invoked_real_tool = any(
        step.tool_name is not None and step.tool_name != "final_answer" and step.error is None
        for step in last_result.steps
    )
    assert invoked_real_tool, (
        f"real local model never successfully invoked a real tool across 5 attempts; "
        f"last trace: {last_result.steps}"
    )
    assert last_result.final_answer is not None
