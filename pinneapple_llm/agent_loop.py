"""A genuine multi-step LLM agent loop: the LLM itself decides, turn by
turn, which real tool to call next, observes the result, and decides the
next step -- grounded in :func:`pinneapple_llm._dispatch.call_llm` (a
real, single-call LLM dispatch across anthropic/openai/ollama) plus a
real tool registry (:class:`pinneapple_worldmodel.physics_tools.PhysicsToolRegistry`
by default, or any object exposing the same ``get``/``available_tools``
query interface).

This is deliberately NOT the same thing as
:func:`pinneapple_problemdesign.knowledge.mapping.available_orchestrator_tools`
(a read-only, single-shot bridge used only to *list* tools for plan
generation) nor :class:`pinneapple_worldmodel.orchestrator.PhysicsOrchestrator`
(a hand-written, goal-directed tool *chain* -- its own planning logic
decides what runs next, not an LLM). Here, each step's tool choice is a
live LLM decision: the loop builds a prompt naming the goal, the history
of what has happened so far, and the tools currently available (queried
live from the registry, never hardcoded), asks the LLM to respond with
either a tool call or a final answer, actually invokes the chosen tool
via the registry, and feeds the real result back into the next turn's
prompt.

JSON-response parsing mirrors the house pattern used by
``pinneapple_problemdesign/extractor.py`` (``_strip_code_fences`` /
``_extract_first_json_object``) rather than importing it -- this module
depends only on ``pinneapple_llm`` and (optionally, lazily)
``pinneapple_worldmodel``, never on ``pinneapple_problemdesign``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ._dispatch import call_llm

__all__ = ["AgentStep", "AgentLoopResult", "run_agent_loop"]


_SYSTEM_PROMPT = """
You are a physics-AI agent that solves goals by calling real tools, one
step at a time.

CRITICAL RULES:
- At each turn you may either call ONE tool, or declare a final answer.
- Only use tools from the "AVAILABLE TOOLS" list below -- never invent a
  tool name or an argument that isn't listed for it.
- Return VALID JSON only. No markdown, no code fences, no commentary
  outside the JSON object.

To call a tool, respond with exactly:
{"action": "call_tool", "tool_name": "<name>", "tool_args": {...}, "reasoning": "<why this tool, briefly>"}

To declare you are done, respond with exactly:
{"action": "final_answer", "answer": "<your final answer>", "reasoning": "<why you're done, briefly>"}
""".strip()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One turn of the loop: the tool the LLM chose (or ``None`` for a
    final-answer / error turn that called no tool), the arguments it
    supplied, the real result of invoking that tool (or ``error`` set
    instead when invocation failed), and the LLM's stated reasoning for
    the turn, if the provider/prompt elicited one."""
    tool_name: Optional[str]
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    error: Optional[str] = None
    llm_reasoning: Optional[str] = None


@dataclass
class AgentLoopResult:
    """Full trace of a :func:`run_agent_loop` run."""
    steps: List[AgentStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    stopped_reason: str = "max_steps"  # "final_answer" | "max_steps" | "error"


# ---------------------------------------------------------------------------
# JSON-response parsing -- mirrors extractor.py's house pattern
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """If the model returns extra text, try to extract the first JSON object."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text  # fallback: unbalanced, let json.loads report the error


def _parse_agent_response(raw_text: str) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
    """Returns ``(parsed, None)`` on success or ``(None, error_message)``
    on failure. Never raises."""
    cleaned = _strip_code_fences(raw_text or "")
    cleaned = _extract_first_json_object(cleaned)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        return None, f"invalid JSON ({exc}); raw response: {raw_text!r}"
    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}: {raw_text!r}"
    return parsed, None


# ---------------------------------------------------------------------------
# Tool-registry introspection
# ---------------------------------------------------------------------------

def _describe_available_tools(tool_registry: Any) -> List[Dict[str, Any]]:
    """Pull live tool metadata (name/category/description/expected args)
    from *tool_registry*. Works with any object exposing an
    ``available_tools()`` method returning objects with
    ``name``/``category``/``description``/``input_schema`` attributes
    (i.e. the same shape as ``PhysicsToolRegistry.available_tools()``),
    so this loop is not hard-coupled to that one implementation."""
    tools = tool_registry.available_tools()
    described = []
    for tool in tools:
        described.append({
            "name": getattr(tool, "name", None),
            "category": getattr(tool, "category", None),
            "description": getattr(tool, "description", ""),
            "expected_args": getattr(tool, "input_schema", {}) or {},
        })
    return described


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _format_history(steps: List[AgentStep]) -> str:
    if not steps:
        return "(no steps taken yet)"
    lines = []
    for i, step in enumerate(steps, start=1):
        if step.tool_name is None:
            lines.append(f"Step {i}: no tool called. error={step.error!r}")
            continue
        result_repr = repr(step.tool_result)
        if len(result_repr) > 500:
            result_repr = result_repr[:500] + "...<truncated>"
        lines.append(
            f"Step {i}: called tool={step.tool_name!r} args={step.tool_args!r} -> "
            f"result={result_repr} error={step.error!r}"
        )
    return "\n".join(lines)


def _build_prompt(goal: str, steps: List[AgentStep], tools: List[Dict[str, Any]]) -> str:
    tools_json = json.dumps(tools, indent=2, default=str)
    return f"""
GOAL:
{goal}

AVAILABLE TOOLS (live, currently invocable -- do not use any other name):
{tools_json}

HISTORY OF STEPS TAKEN SO FAR:
{_format_history(steps)}

Decide the next step. Respond with exactly one JSON object as described
in the system prompt: either a "call_tool" action naming one tool from
AVAILABLE TOOLS with its arguments, or a "final_answer" action if the
goal has been achieved (or cannot be achieved with the available tools).
""".strip()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    goal: str,
    tool_registry: Any = None,
    *,
    llm_provider: str = "anthropic",
    model: Optional[str] = None,
    max_steps: int = 8,
    llm_call_fn: Optional[Callable[..., str]] = None,
    conversation_store: Any = None,
    **call_llm_kwargs: Any,
) -> AgentLoopResult:
    """Run a genuine multi-step agent loop toward *goal*.

    Each turn: build a prompt listing the goal, prior-step history, and
    the tools currently available live from *tool_registry*; ask the LLM
    to either pick one tool + arguments or declare a final answer; if a
    tool was chosen, actually invoke it via the registry (any exception
    is caught and recorded on the step rather than crashing the loop);
    stop when the LLM declares a final answer, ``max_steps`` is reached,
    or the LLM call itself fails unrecoverably (e.g. no backend
    reachable/configured).

    Parameters
    ----------
    tool_registry : defaults to a real, freshly constructed and
        registered ``PhysicsToolRegistry()`` if not given. Accepts any
        object exposing ``available_tools()`` and ``get(name)`` (the
        same query interface), so it is not hard-coupled to that one
        registry.
    llm_call_fn : optional override of the shape
        ``(prompt: str, *, system: str) -> str``, used instead of
        :func:`pinneapple_llm._dispatch.call_llm`. This is what makes
        the loop's prompt-building / parsing / tool-invocation logic
        deterministically testable without a real LLM backend, and is
        equally useful for callers who want to route through their own
        dispatch (e.g. with custom retries or a different provider).
    **call_llm_kwargs : forwarded to ``call_llm`` when *llm_call_fn* is
        not given (e.g. ``api_key``, ``host``/``port`` for ``"ollama"``).
    """
    if tool_registry is None:
        from pinneapple_worldmodel.physics_tools import PhysicsToolRegistry

        tool_registry = PhysicsToolRegistry()
        tool_registry.register_all()

    if llm_call_fn is None:
        def llm_call_fn(prompt: str, *, system: str = "") -> str:
            return call_llm(
                prompt,
                provider=llm_provider,
                model=model,
                system=system,
                json_mode=True,
                module="agent_loop",
                conversation_store=conversation_store,
                **call_llm_kwargs,
            )

    steps: List[AgentStep] = []

    for _ in range(max_steps):
        tools = _describe_available_tools(tool_registry)
        prompt = _build_prompt(goal, steps, tools)

        try:
            raw_response = llm_call_fn(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            steps.append(AgentStep(
                tool_name=None, tool_args={}, tool_result=None,
                error=f"LLM call failed: {exc}", llm_reasoning=None,
            ))
            return AgentLoopResult(steps=steps, final_answer=None, stopped_reason="error")

        parsed, parse_error = _parse_agent_response(raw_response)
        if parse_error is not None:
            # A malformed response from the LLM is recoverable: record it
            # and let the LLM see it happened via the growing history on
            # the next turn, rather than aborting the whole loop.
            steps.append(AgentStep(
                tool_name=None, tool_args={}, tool_result=None,
                error=f"Could not parse LLM response as JSON: {parse_error}",
                llm_reasoning=None,
            ))
            continue

        reasoning = parsed.get("reasoning")
        action = parsed.get("action")

        if action == "final_answer":
            answer = parsed.get("answer", "")
            steps.append(AgentStep(
                tool_name="final_answer", tool_args={}, tool_result=answer,
                error=None, llm_reasoning=reasoning,
            ))
            return AgentLoopResult(steps=steps, final_answer=answer, stopped_reason="final_answer")

        if action == "call_tool":
            tool_name = parsed.get("tool_name")
            tool_args = parsed.get("tool_args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}

            if not tool_name:
                steps.append(AgentStep(
                    tool_name=None, tool_args=tool_args, tool_result=None,
                    error="LLM chose action 'call_tool' but supplied no tool_name",
                    llm_reasoning=reasoning,
                ))
                continue

            try:
                tool = tool_registry.get(tool_name)
                result = tool.call(**tool_args)
                steps.append(AgentStep(
                    tool_name=tool_name, tool_args=tool_args, tool_result=result,
                    error=None, llm_reasoning=reasoning,
                ))
            except Exception as exc:
                # A bad tool choice/args is a real, recorded failure --
                # not a crash. The LLM gets to see it and try again.
                steps.append(AgentStep(
                    tool_name=tool_name, tool_args=tool_args, tool_result=None,
                    error=str(exc), llm_reasoning=reasoning,
                ))
            continue

        # Unknown action -- recoverable, same as a parse error.
        steps.append(AgentStep(
            tool_name=None, tool_args={}, tool_result=None,
            error=f"LLM returned unknown action {action!r}", llm_reasoning=reasoning,
        ))

    return AgentLoopResult(steps=steps, final_answer=None, stopped_reason="max_steps")
