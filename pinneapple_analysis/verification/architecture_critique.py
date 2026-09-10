"""Adversarial review of a proposed Physics-AI architecture/pipeline
design -- "assume the role of a Principal Engineer trying to prove this
architecture is wrong" as a structured, checked-menu LLM tool, not free-
form commentary.

Why a fixed checklist instead of an open-ended "critique this" prompt:
the whole point of an adversarial review is to catch failure modes the
architecture's OWN author didn't think to ask about -- an open-ended
prompt tends to just restate whatever the author already emphasized. This
module forces the LLM to produce a verdict for EVERY category in
``FAILURE_MODE_CHECKLIST`` (data leakage, shortcut learning, dimensional
inconsistency, identifiability, extrapolation failure, unstable training,
bad benchmark design, misleading metrics, malformed physics constraints,
excessive simulator/solver dependency, deployment risk) -- the same
"checked menu, reject anything outside it, reject an incomplete response"
anti-hallucination pattern already used throughout this codebase
(``pinneapple_llm.cad_draft``/``geometry_draft``, and this package's own
``geometry_intelligence._llm_assign_semantics``).

This is explicitly a STRUCTURED PROMPT for a general-purpose LLM, not a
new trained classifier -- it does not (and cannot) guarantee the LLM's
critique is correct; it guarantees the critique's SHAPE is complete and
checkable (every category addressed, every verdict/severity value drawn
from a fixed vocabulary, an invented category name or verdict outside
that vocabulary is rejected rather than silently accepted) -- the same
honesty tradeoff every other LLM-facing tool in this codebase makes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Fixed, checked vocabulary -- see module docstring for why this is a
# checklist and not an open-ended prompt. Each entry's description is
# shown to the LLM verbatim so it knows exactly what to check for, not a
# one-word category name it has to interpret on its own.
FAILURE_MODE_CHECKLIST: Dict[str, str] = {
    "data_leakage": (
        "Does any information from the test/validation set (or from the future, in a "
        "time-dependent problem) leak into training -- e.g. normalizing using statistics "
        "computed over the whole dataset before splitting, or collocation points that "
        "coincide with held-out evaluation points?"
    ),
    "shortcut_learning": (
        "Could the model achieve low training/validation loss by exploiting a spurious "
        "correlation in the data rather than the actual physics -- e.g. a nearly-constant "
        "field where predicting the mean already gives a deceptively low MSE?"
    ),
    "dimensional_inconsistency": (
        "Are all terms in the loss (physics residual, data loss, BC loss) on comparable "
        "physical/numerical scales, or could one term dominate purely due to unit/magnitude "
        "mismatch rather than genuine relative importance?"
    ),
    "identifiability_problem": (
        "If any physical parameter is being estimated (an inverse problem), is there enough "
        "independent information in the observations to actually pin it down -- or could "
        "multiple different parameter values explain the data equally well?"
    ),
    "extrapolation_failure": (
        "Will the trained model be queried on parameters/geometries/regimes outside its "
        "training distribution, and if so, is there any mechanism (UQ, OOD detection, a "
        "documented validity range) to flag that rather than silently extrapolating?"
    ),
    "unstable_training": (
        "Are there known instability risks for this architecture/problem combination -- "
        "stiff PDE residual gradients, adversarial loss-term competition, L-BFGS history-size "
        "growth on long single phases, vanishing/exploding gradients in a deep/recurrent "
        "component -- and is there a concrete mitigation, not just hope?"
    ),
    "bad_benchmark_design": (
        "Does the evaluation protocol actually test what's claimed -- e.g. is the test set "
        "drawn from a genuinely different distribution than training when generalization is "
        "the claim, or is 'accuracy' measured only at training-adjacent points?"
    ),
    "misleading_metrics": (
        "Could the headline metric (e.g. a pooled/averaged residual or RMSE) hide a real "
        "failure in a specific region/regime -- e.g. an average that looks good because most "
        "of the domain is trivially easy, masking a bad fit where it matters most?"
    ),
    "malformed_physics_constraints": (
        "Are the boundary/initial conditions, conservation laws, and governing equation "
        "actually correct for the stated physical problem -- not just syntactically valid "
        "code, but the right physics?"
    ),
    "deployment_dependency_risk": (
        "Does this architecture/pipeline depend on external solvers, proprietary licenses, "
        "specific hardware (GPU-only), or services that could become unavailable, and is "
        "there a fallback, or is this an unstated single point of failure?"
    ),
}

_ALLOWED_VERDICTS = {"concern", "not_applicable", "no_issue_found"}
_ALLOWED_SEVERITIES = {"low", "medium", "high"}


@dataclass
class CritiqueFinding:
    category: str  # one of FAILURE_MODE_CHECKLIST's keys
    verdict: str  # "concern" | "not_applicable" | "no_issue_found"
    severity: Optional[str]  # "low" | "medium" | "high" -- required iff verdict == "concern"
    reasoning: str


@dataclass
class AdversarialReviewReport:
    architecture_description: str
    findings: List[CritiqueFinding]
    overall_reasoning: str
    provider: str
    raw_llm_response: str = ""

    @property
    def concerns(self) -> List[CritiqueFinding]:
        return [f for f in self.findings if f.verdict == "concern"]

    @property
    def high_severity_concerns(self) -> List[CritiqueFinding]:
        return [f for f in self.concerns if f.severity == "high"]


_SYSTEM_PROMPT = """You are a skeptical Principal Engineer performing an adversarial review of \
a proposed Physics-AI architecture/pipeline. Your job is to find real reasons this design might \
fail, not to praise it -- an architecture that "looks impressive" can still be fundamentally wrong.

You are given a FIXED checklist of failure-mode categories, each with a description of exactly \
what to check for. You MUST address EVERY category -- do not invent new categories, do not skip \
any, do not merge two categories into one entry.

You MUST respond with a single JSON object and nothing else, of the exact form:
{"findings": [{"category": "<one of the exact category keys given>", \
"verdict": "<one of: concern | not_applicable | no_issue_found>", \
"severity": "<one of: low | medium | high, OR null if verdict is not \"concern\">", \
"reasoning": "<one or two sentences, specific to THIS architecture, not generic advice>"}], \
"overall_reasoning": "<two or three sentences summarizing the most important concerns, or why \
none were found>"}

Rules:
- "category" MUST be exactly one of the category keys given to you. Never invent a new category.
- Provide EXACTLY one finding per category given, no more, no fewer.
- "verdict" must be "concern" only when you have a SPECIFIC, architecture-relevant reason -- use \
"not_applicable" when the category genuinely doesn't apply to this architecture (e.g. \
identifiability_problem for a pure forward problem with no parameter estimation), and \
"no_issue_found" when the category applies but you see no specific problem.
- "severity" MUST be null unless verdict is "concern" -- never assign a severity to a non-concern.
"""


def _build_prompt(architecture_description: str, spec_context: str) -> str:
    checklist_text = "\n".join(
        f"- {key}: {desc}" for key, desc in FAILURE_MODE_CHECKLIST.items()
    )
    return (
        f"ARCHITECTURE/PIPELINE DESCRIPTION:\n{architecture_description}\n\n"
        f"{spec_context}"
        f"CHECKLIST (address every one of these {len(FAILURE_MODE_CHECKLIST)} categories):\n"
        f"{checklist_text}\n"
    )


def run_adversarial_review(
    architecture_description: str,
    spec: Any = None,
    *,
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    conversation_store: Any = None,
) -> AdversarialReviewReport:
    """Run a structured adversarial review of *architecture_description*
    (free-text: the proposed model/pipeline design, e.g. "FNO surrogate
    trained on 200 OpenFOAM runs, fine-tuned with a physics residual
    loss, deployed for real-time inference") against the fixed
    :data:`FAILURE_MODE_CHECKLIST`. See the module docstring for exactly
    what this checked-menu design does and does not guarantee.

    Parameters
    ----------
    architecture_description : the design to critique, in plain text.
    spec : optional ``ProblemSpec`` -- if given, its ``pde.kind``/coords/
        domain_bounds are included as context so the review is grounded
        in the actual physics, not just the architecture description
        alone.
    provider / model / api_key / conversation_store : forwarded to
        ``pinneapple_llm.call_llm`` unchanged.

    Raises
    ------
    ValueError
        The LLM response is not valid JSON, is missing a category, names
        a category/verdict/severity outside the fixed vocabulary, or
        assigns a non-null severity to a non-"concern" verdict -- a
        hallucinated or malformed response is rejected, never silently
        accepted (the same discipline as every other checked-menu tool
        in this codebase).
    """
    import pinneapple_llm as pl

    spec_context = ""
    if spec is not None:
        kind = getattr(getattr(spec, "pde", None), "kind", None)
        coords = getattr(spec, "coords", None)
        bounds = getattr(spec, "domain_bounds", None)
        spec_context = (
            f"PHYSICS CONTEXT: pde.kind={kind!r}, coords={coords!r}, "
            f"domain_bounds={bounds!r}\n\n"
        )

    prompt = _build_prompt(architecture_description, spec_context)
    raw = pl.call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="architecture_critique", conversation_store=conversation_store,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    raw_findings = parsed.get("findings")
    overall_reasoning = parsed.get("overall_reasoning", "")
    if not isinstance(raw_findings, list):
        raise ValueError(f"LLM response missing a JSON list under 'findings': {raw}")

    seen_categories = set()
    findings: List[CritiqueFinding] = []
    for entry in raw_findings:
        if not isinstance(entry, dict):
            raise ValueError(f"LLM finding entry is not a JSON object: {entry!r}")
        category = entry.get("category")
        if category not in FAILURE_MODE_CHECKLIST:
            raise ValueError(
                f"LLM named category {category!r}, not in the real checklist "
                f"{sorted(FAILURE_MODE_CHECKLIST)} -- refusing a hallucinated category."
            )
        verdict = entry.get("verdict")
        if verdict not in _ALLOWED_VERDICTS:
            raise ValueError(
                f"LLM proposed verdict {verdict!r} for category {category}, not in the real "
                f"allowed set {sorted(_ALLOWED_VERDICTS)} -- refusing a hallucinated verdict."
            )
        severity = entry.get("severity")
        if verdict == "concern":
            if severity not in _ALLOWED_SEVERITIES:
                raise ValueError(
                    f"LLM assigned verdict='concern' for category {category} but severity "
                    f"{severity!r} is not in {sorted(_ALLOWED_SEVERITIES)}."
                )
        elif severity is not None:
            raise ValueError(
                f"LLM assigned a non-null severity {severity!r} to category {category} with "
                f"verdict={verdict!r} -- severity must be null unless verdict is 'concern'."
            )
        reasoning = entry.get("reasoning", "")
        seen_categories.add(category)
        findings.append(CritiqueFinding(category=category, verdict=verdict, severity=severity,
                                         reasoning=reasoning))

    missing = set(FAILURE_MODE_CHECKLIST) - seen_categories
    if missing:
        raise ValueError(f"LLM response did not address every checklist category -- missing {sorted(missing)}.")

    return AdversarialReviewReport(
        architecture_description=architecture_description, findings=findings,
        overall_reasoning=overall_reasoning, provider=provider, raw_llm_response=raw,
    )
