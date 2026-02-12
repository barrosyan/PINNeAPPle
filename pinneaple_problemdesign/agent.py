from __future__ import annotations

from typing import Any, Dict, List, Optional

from .protocol import LLMProvider, LLMMessage
from .state import DesignState
from .schema import Gap
from .extractor import extract_from_text
from .merge import merge_into_spec
from .policy import apply_non_invention_policy

from .elicitation.stages import STAGES_ORDER
from .elicitation.questions import build_stage_gaps
from .elicitation.validators import validate_and_suggest

from .knowledge.mapping import build_plan_fno_first
from .renderers.report_md import render_markdown_report


_EMPTY = (None, "", [], {})


def _deep_get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _stage_required_satisfied(spec_dict: Dict[str, Any], required_fields: List[str]) -> bool:
    for f in required_fields:
        val = _deep_get(spec_dict, f) if "." in f else spec_dict.get(f)
        if val in _EMPTY:
            return False
    return True


def _dedup_gaps(gaps: List[Gap]) -> List[Gap]:
    out: List[Gap] = []
    seen = set()
    for g in gaps:
        if g.id in seen:
            continue
        seen.add(g.id)
        out.append(g)
    return out


class DesignAgent:
    """
    High-level flow:
      - ingest user message
      - structured extraction (json_mode=True)
      - safe merge into ProblemSpec
      - non-invention policy => unknown -> Gap, assumptions -> Assumption
      - stage-based gap generation
      - once sufficient => build plan (FNO-first) and render report
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def start(self) -> DesignState:
        return DesignState()

    def ingest_user_message(self, state: DesignState, user_text: str) -> None:
        state.add_turn("user", user_text)

        # 1) Structured extraction (no guessing)
        partial, meta = extract_from_text(self.llm, user_text)

        # 2) Safe merge into spec
        merge_into_spec(state.spec, partial)

        # 3) Apply non-invention policy
        apply_non_invention_policy(
            state,
            unknown_fields=meta.get("unknown_fields", []),
            assumptions_suggested=meta.get("assumptions_suggested", []),
            gaps_suggested=meta.get("gaps_suggested", []),
        )

    def step(self, state: DesignState) -> Dict[str, Any]:
        if state.done:
            report = state.to_report()
            warnings, _ = validate_and_suggest(report.spec)
            md = render_markdown_report(report, warnings)
            return {"type": "report", "markdown": md, "report": report}

        # Validation + add suggested assumptions (still needs confirmation)
        warnings, assumptions = validate_and_suggest(state.spec)
        for a in assumptions:
            state.spec.assumptions.append(a)

        # Stage gaps (heuristic) — these are safe because they ask, not assume
        spec_dict = state.spec.to_dict()
        stage_gaps = build_stage_gaps(state.stage, spec_dict)

        # Merge stage gaps without duplication
        existing = {g.id: g for g in state.gaps}
        for g in stage_gaps:
            if g.id not in existing:
                state.gaps.append(g)

        state.gaps = _dedup_gaps(state.gaps)

        # Advance stage if required fields are satisfied
        stage_obj = next(s for s in STAGES_ORDER if s.name == state.stage)
        if _stage_required_satisfied(spec_dict, stage_obj.required_fields):
            state.stage = self._next_stage(state.stage)

            if state.stage == "finalization":
                # Build the plan + finalize
                state.plan = build_plan_fno_first(state.spec, state.gaps)
                state.done = True
                report = state.to_report()
                warnings2, _ = validate_and_suggest(report.spec)
                md = render_markdown_report(report, warnings2)
                return {"type": "report", "markdown": md, "report": report}

        # Pick a small batch of pending questions (prioritize blocker)
        pending = [g for g in state.gaps if not g.resolved]
        pending_sorted = sorted(
            pending,
            key=lambda gg: {"blocker": 0, "important": 1, "nice_to_have": 2}.get(gg.severity, 9),
        )
        batch = pending_sorted[:5]

        # Optional: rewrite as crisp numbered questions using LLM (not required to work)
        questions_text = self._rewrite_questions(state, batch)

        return {
            "type": "questions",
            "stage": state.stage,
            "questions": batch,
            "questions_text": questions_text,
            "warnings": warnings,
        }

    def resolve_gap(self, state: DesignState, gap_id: str) -> None:
        for g in state.gaps:
            if g.id == gap_id:
                g.resolved = True
                return

    def _next_stage(self, current: str) -> str:
        names = [s.name for s in STAGES_ORDER]
        i = names.index(current)
        if i >= len(names) - 1:
            return "finalization"
        return names[i + 1]

    def _rewrite_questions(self, state: DesignState, gaps: List[Gap]) -> str:
        if not gaps:
            return "No questions."

        # If you ever want to disable LLM rewriting, just return a simple list here.
        prompt = [
            LLMMessage(
                role="system",
                content=(
                    "You are an engineering assistant. Convert the list into short, non-redundant, "
                    "numbered questions. Do not add new questions. Do not invent facts."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Context:\n- title: {state.spec.title}\n- goal: {state.spec.goal}\n- task: {state.spec.task_type}\n\n"
                    "Questions to rewrite:\n" + "\n".join([f"- [{g.severity}] {g.question}" for g in gaps])
                ),
            ),
        ]
        try:
            resp = self.llm.generate(prompt, temperature=0.2, max_tokens=250, json_mode=False)
            return resp.text
        except Exception:
            return "\n".join([f"{i+1}. {g.question}" for i, g in enumerate(gaps)])
