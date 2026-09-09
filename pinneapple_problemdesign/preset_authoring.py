"""LLM-assisted preset *drafting*, gated by a real, automatically-generated
self-consistency check -- not a preset *authoring pipeline* in the sense of
producing something trusted for production use.

Why this module exists
-----------------------
Every physics domain PINNeAPPle can currently solve needs a human to
hand-author a preset in ``pinneapple_physics.pde_environment.presets``
ahead of time. Full automatic physics discovery (an LLM correctly inventing
a governing equation for a genuinely new phenomenon, unaided) is a real,
open research problem and is explicitly *out of scope* here. What this
module does instead is the safe half of that idea:

1. Ask an LLM to *propose* a governing PDE (plus BCs/ICs) for a plain-
   English phenomenon description, grounded in real, citable equations
   already in this repo's knowledge base
   (``pinneapple_problemdesign.knowledge.physics_knowledge``) wherever a
   relevant entry exists.
2. NEVER trust that proposal blindly. Parse it into a real ``sympy``
   expression (a hard failure, not a silent fallback, if it doesn't
   parse), compile it with the real symbolic-PDE machinery
   (``pinneapple_physics.symbolic_pde``), and run a real method-of-
   manufactured-solutions (MMS) check against it -- the exact technique
   ``tests/test_manufactured_solutions.py`` uses for this repo's real,
   human-authored presets: plug a known closed-form exact solution into
   the compiled residual (using real autograd derivatives) and confirm
   the residual is close to zero.

This turns "LLM invents physics" (dangerous, prone to hallucination) into
"LLM proposes, code verifies" (safe, grounded) -- but the verification is
deliberately blunt, not a physics oracle. Grounding the prompt in real
knowledge-base citations measurably *reduces* hallucination risk (the LLM
is more likely to reuse or lightly adapt a real, working equation instead
of inventing one from nothing); it does **not eliminate** it, and the
MMS check catches only *self-inconsistency* (does the proposed equation,
with its own stated parameters, actually admit a simple textbook-style
exact solution), not "is this the correct governing physics of the real
phenomenon described". A `DraftedPreset` with ``is_verified=True`` means
"this equation is not obviously broken", not "this equation is right".

A verified `DraftedPreset` is data. This module deliberately does **not**
register it into the real ``pinneapple_physics.pde_environment.presets``
registry -- promoting a draft to a real, trusted preset file is left as a
human (or a separately-built, more conservative research-loop) decision.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp
import torch
import torch.nn as nn

from pinneapple_llm._dispatch import call_llm
from pinneapple_physics.symbolic_pde.compiler import pde_from_sympy
from pinneapple_problemdesign.knowledge.physics_knowledge import (
    PhenomenonEntry,
    lookup_phenomenon,
)

__all__ = [
    "PresetDraftError",
    "ManufacturedSolutionCheck",
    "DraftedPreset",
    "run_manufactured_solution_check",
    "draft_preset",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PresetDraftError(Exception):
    """Raised when the LLM's proposal cannot be turned into a real,
    checkable PDE.

    This module never silently substitutes a fallback equation for a
    proposal it cannot parse or make sense of -- that would be exactly
    the kind of silent invention this module exists to prevent. Every
    such failure is a hard exception, not a degraded/partial result.

    Attributes
    ----------
    stage : which step failed (``"llm_json_parse"``, ``"missing_fields"``,
        ``"invalid_name"``, ``"pde_parse"``, ``"undeclared_field"``,
        ``"missing_param_value"``) -- structured, so a caller (e.g. a
        research loop) can log/branch on *why* a draft failed to even
        become checkable, distinct from *failing its check*.
    message : human-readable detail.
    raw : the raw LLM response text (or offending substring), kept for
        audit, same reasoning as ``proposed_equation_latex_or_sympy`` on
        `DraftedPreset`.
    """

    def __init__(self, stage: str, message: str, *, raw: str = "") -> None:
        self.stage = stage
        self.message = message
        self.raw = raw
        super().__init__(f"[{stage}] {message}")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ManufacturedSolutionCheck:
    """One method-of-manufactured-solutions self-consistency check.

    Mirrors ``tests/test_manufactured_solutions.py``'s technique: an exact
    closed-form function is substituted for the field(s), the compiled PDE
    residual is evaluated at real collocation points via real autograd
    derivatives, and the (root-mean-square) residual magnitude is compared
    against ``tolerance``.

    Attributes
    ----------
    exact_solution_used : mapping of field name -> the exact-solution
        expression (as a string) plugged in for that field. All fields
        currently share the same closed-form profile (see
        `_select_exact_solution` for why) -- a known coarse limitation for
        genuinely coupled multi-field systems (e.g. Navier-Stokes), noted
        here rather than hidden.
    residual_at_exact_solution : the real computed root-mean-square PDE
        residual over the sampled collocation points. ~0 means the
        compiled PDE is self-consistent with this exact solution; a large
        value does NOT necessarily mean the physics is wrong -- it may
        simply mean this particular simple candidate isn't an exact
        solution of this particular equation (see module docstring).
    tolerance : the pass/fail threshold applied to
        ``residual_at_exact_solution``.
    passed : ``residual_at_exact_solution < tolerance``.
    n_points : number of collocation points sampled.
    notes : short human-readable summary, useful when logging a failed
        check without re-deriving the numbers.
    """

    exact_solution_used: Dict[str, str]
    residual_at_exact_solution: float
    tolerance: float
    passed: bool
    n_points: int = 0
    notes: str = ""


@dataclass
class DraftedPreset:
    """An LLM-proposed PDE preset, together with the real verification
    that was run against it. See module docstring for the safety model.

    Attributes
    ----------
    phenomenon_description : the original plain-English request.
    proposed_equation_latex_or_sympy : the LLM's raw proposed-equation
        string, kept verbatim for audit (this is what a human reviews to
        judge "did the LLM propose something sane", independent of
        whether it parsed/verified).
    sympy_pde : the REAL, successfully parsed ``sympy`` expression built
        from the proposal (residual form, expected == 0). Never a
        fallback/placeholder -- if parsing fails, `draft_preset` raises
        `PresetDraftError` instead of constructing a `DraftedPreset`.
    proposed_bcs : the LLM's proposed boundary conditions (free-text /
        structured dicts, not independently verified by this module --
        only the PDE residual is checked, matching
        ``test_manufactured_solutions.py``'s scope).
    proposed_ics : the LLM's proposed initial conditions (same caveat).
    verification_result : the real `ManufacturedSolutionCheck` that was
        run against `sympy_pde`.
    is_verified : ``verification_result.passed``.
    citations_used : which `physics_knowledge` entries (as
        ``"module.function"`` strings) were included as grounding context
        in the LLM prompt, if any -- empty if `use_knowledge_base=False`
        or nothing matched.
    coords : coordinate symbol names used to build `sympy_pde`.
    fields : field names used to build `sympy_pde`.
    params : numeric parameter values used to build `sympy_pde` (and fed
        into the manufactured-solution check).
    raw_llm_response : the raw LLM response text, kept for audit.
    """

    phenomenon_description: str
    proposed_equation_latex_or_sympy: str
    sympy_pde: sp.Expr
    proposed_bcs: List[Any]
    proposed_ics: List[Any]
    verification_result: ManufacturedSolutionCheck
    is_verified: bool
    citations_used: List[str] = field(default_factory=list)
    coords: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    raw_llm_response: str = ""


# ---------------------------------------------------------------------------
# JSON extraction from the LLM response -- mirrors
# pinneapple_problemdesign.extractor's `_strip_code_fences` /
# `_extract_first_json_object` house style (copied rather than imported so
# this module stays self-contained; the logic is intentionally identical).
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
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
    return text


# ---------------------------------------------------------------------------
# Knowledge-base grounding
# ---------------------------------------------------------------------------

def _grounding_hits(description: str, max_hits: int = 5) -> List[PhenomenonEntry]:
    """Look up real knowledge-base entries relevant to a free-text
    description. `lookup_phenomenon` matches by substring, so a full
    sentence rarely matches directly -- fall back to trying individual
    "significant" (length > 4) words from the description if the whole
    string finds nothing."""
    hits = lookup_phenomenon(description)
    if not hits:
        seen: List[PhenomenonEntry] = []
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", description):
            for entry in lookup_phenomenon(word):
                if entry not in seen:
                    seen.append(entry)
        hits = seen
    return hits[:max_hits]


def _format_grounding(hits: Sequence[PhenomenonEntry]) -> str:
    if not hits:
        return "(no matching entries found in the physics knowledge base)"
    payload = [
        {
            "phenomenon": h.phenomenon,
            "governing_equation": h.governing_equation,
            "typical_parameters": h.typical_parameters,
            "assumptions": list(h.assumptions),
            "source": f"{h.preset_module}.{h.preset_function}",
        }
        for h in hits
    ]
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a computational-physics modeling assistant helping draft a governing
PDE for a new problem. Your proposal will NOT be trusted blindly -- it will
be parsed into a real symbolic expression and checked against a real
manufactured-solution residual test before anyone uses it. Because of that:

CRITICAL RULES:
- Prefer reusing or lightly adapting a REAL, KNOWN governing equation for
  the described phenomenon (cite it via "grounded_in") over inventing one
  from scratch. If real grounding context is provided below, use it.
- Express the PDE as a single sympy-parseable RESIDUAL expression equal to
  zero (do not include "= 0", just the left-hand side after moving
  everything to one side).
- Fields MUST be written as function applications over ALL declared
  coordinates, e.g. a field "u" over coords ["x","t"] must appear as
  "u(x, t)" everywhere, including inside derivatives.
- Derivatives MUST use sympy's Derivative(...) syntax, e.g.
  "Derivative(u(x, t), t)" for du/dt, "Derivative(u(x, t), x, 2)" for
  d^2u/dx^2.
- You may use: sin, cos, tan, exp, log, sqrt, pi, E, Abs, sign, tanh,
  sinh, cosh, and named coordinates/fields/parameters. Nothing else.
- Do NOT introduce a coordinate, field, or parameter name that isn't
  declared in "coords"/"fields"/"params" respectively.
- Every parameter symbol used in the equation MUST have a numeric default
  value in "params".
- Return VALID JSON only. No markdown. No commentary.
""".strip()


_USER_TEMPLATE = """
PHENOMENON DESCRIPTION:
<<DESCRIPTION>>

REAL GROUNDING CONTEXT (from this codebase's physics knowledge base --
cite by "source" in "grounded_in" if you use one, empty list if you don't):
<<GROUNDING>>

Return a JSON object with exactly these keys:
- "coords": list of coordinate names, e.g. ["x", "t"].
- "fields": list of dependent-field names, e.g. ["u"].
- "params": object mapping parameter name -> numeric default value, e.g.
  {"D": 1.0}.
- "pde_residual": the PDE residual string, per the rules above.
- "boundary_conditions": list of short free-text boundary-condition
  descriptions.
- "initial_conditions": list of short free-text initial-condition
  descriptions (empty list if the problem is steady-state).
- "grounded_in": list of "source" strings (from the grounding context)
  this proposal was adapted from, empty list if none applied.
- "reasoning": one or two sentences explaining the choice.

Example (1D transient heat conduction with a fixed sinusoidal source):
{
  "coords": ["x", "t"],
  "fields": ["u"],
  "params": {"alpha": 1.0, "Q": 0.5},
  "pde_residual": "Derivative(u(x, t), t) - alpha*Derivative(u(x, t), x, 2) - Q*sin(pi*x)",
  "boundary_conditions": ["u(0,t) = 0", "u(1,t) = 0"],
  "initial_conditions": ["u(x,0) = 0"],
  "grounded_in": ["pinneapple_physics.pde_environment.presets.engineering.car_brake_thermal"],
  "reasoning": "Standard 1D transient heat equation with an explicit sinusoidal source term."
}
""".strip()


def _build_prompt(description: str, grounding: str) -> str:
    return _USER_TEMPLATE.replace("<<DESCRIPTION>>", description).replace("<<GROUNDING>>", grounding)


# ---------------------------------------------------------------------------
# Parsing the LLM's proposal into a real sympy expression
# ---------------------------------------------------------------------------

_MATH_NAMESPACE = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "pi": sp.pi, "E": sp.E, "Abs": sp.Abs, "sign": sp.sign,
    "tanh": sp.tanh, "sinh": sp.sinh, "cosh": sp.cosh,
    "Derivative": sp.Derivative,
}


def _normalize_equation_string(s: str) -> str:
    """If the LLM ignored the "residual only" instruction and returned a
    "lhs = rhs" equation, rewrite it as "(lhs) - (rhs)". Skips ==, <=, >=,
    != so it doesn't mangle a comparison the LLM might (wrongly) include."""
    s = s.strip()
    m = re.search(r"(?<![=<>!])=(?!=)", s)
    if not m:
        return s
    lhs, rhs = s[: m.start()], s[m.start() + 1 :]
    return f"({lhs}) - ({rhs})"


def _parse_proposed_pde(
    pde_str: str,
    coord_names: Sequence[str],
    field_names: Sequence[str],
    param_values: Dict[str, Any],
) -> Tuple[sp.Expr, List[sp.Symbol], List[sp.Function], List[sp.Symbol], Dict[str, float]]:
    """Parse `pde_str` into a real sympy expression, hard-failing with a
    structured `PresetDraftError` on anything that isn't a clean,
    self-contained proposal. Returns (expr, coord_syms, field_syms,
    param_syms, resolved_params)."""
    for kind, names in (("coordinate", coord_names), ("field", field_names)):
        for name in names:
            if not isinstance(name, str) or not name.isidentifier():
                raise PresetDraftError(
                    "invalid_name", f"LLM proposed an invalid {kind} name {name!r} (not a valid Python identifier)."
                )

    coord_syms = [sp.Symbol(n) for n in coord_names]
    field_syms = [sp.Function(n) for n in field_names]
    coord_name_set = set(coord_names)
    field_name_set = set(field_names)

    namespace: Dict[str, Any] = dict(_MATH_NAMESPACE)
    namespace.update({n: s for n, s in zip(coord_names, coord_syms)})
    namespace.update({n: f for n, f in zip(field_names, field_syms)})
    # Any parameter the LLM declared a value for is made available by name
    # too, so e.g. "D" in the equation resolves to a Symbol("D") rather
    # than sympy silently treating it as a brand-new free symbol.
    param_syms_from_payload = {n: sp.Symbol(n) for n in param_values if n not in coord_name_set and n not in field_name_set}
    namespace.update(param_syms_from_payload)

    normalized = _normalize_equation_string(pde_str)
    try:
        expr = sp.sympify(normalized, locals=namespace)
    except Exception as e:
        raise PresetDraftError(
            "pde_parse", f"Could not parse proposed PDE residual as sympy: {e}", raw=pde_str
        ) from e

    if not isinstance(expr, sp.Basic):
        raise PresetDraftError("pde_parse", f"Parsed result is not a sympy expression: {expr!r}", raw=pde_str)

    # Any AppliedUndef function in the parsed expression must be one of
    # the declared fields -- an LLM writing e.g. "v(x,t)" without
    # declaring "v" in "fields" is an inconsistent proposal, not something
    # to silently accept.
    undeclared_funcs = set()
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.core.function.AppliedUndef):
            fname = sub.func.__name__
            if fname not in field_name_set:
                undeclared_funcs.add(fname)
    if undeclared_funcs:
        raise PresetDraftError(
            "undeclared_field",
            f"Proposed equation uses field(s) {sorted(undeclared_funcs)} not declared in 'fields' {list(field_names)}.",
            raw=pde_str,
        )

    # Any remaining free Symbol that isn't a coordinate is treated as a
    # parameter and MUST have a numeric value supplied -- never guessed.
    free_symbol_names = {str(s) for s in expr.free_symbols}
    implied_param_names = sorted(free_symbol_names - coord_name_set)
    missing = [n for n in implied_param_names if n not in param_values]
    if missing:
        raise PresetDraftError(
            "missing_param_value",
            f"Proposed equation uses parameter(s) {missing} with no numeric value in 'params'.",
            raw=pde_str,
        )

    resolved_params: Dict[str, float] = {}
    for n in implied_param_names:
        try:
            resolved_params[n] = float(param_values[n])
        except (TypeError, ValueError) as e:
            raise PresetDraftError(
                "missing_param_value", f"Parameter '{n}' value {param_values[n]!r} is not numeric.", raw=pde_str
            ) from e

    param_syms = [sp.Symbol(n) for n in implied_param_names]
    return expr, coord_syms, field_syms, param_syms, resolved_params


# ---------------------------------------------------------------------------
# The manufactured-solution check
# ---------------------------------------------------------------------------

class _ExactFn(nn.Module):
    """Wraps a plain torch-differentiable function as a fake "model", same
    pattern as ``tests/test_manufactured_solutions.py``'s helper of the
    same name -- ``SymbolicPDE.to_residual_fn`` calls ``model(x)`` and (via
    the compiler's normal usage) expects an ``nn.Module``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x)


_TIME_COORD_NAMES = {"t", "time"}
_DOMAIN_LOW, _DOMAIN_HIGH = 0.2, 1.2  # avoids the origin (log/1/r singularities)


def _select_exact_solution(coord_syms: Sequence[sp.Symbol]) -> sp.Expr:
    """Explicit, table-driven pick of a simple closed-form exact solution,
    based only on how many spatial coordinates there are and whether one
    of them looks like time. Deliberately NOT "clever": it never inspects
    the equation itself or solves for a matching rate/parameter (that
    would be fragile -- see module docstring on why residual!=0 doesn't
    imply broken physics). The rule:

    - 1 spatial coordinate: sin(pi * x)  (nonzero curvature, simple).
    - >=2 spatial coordinates: x0^2 - x1^2  (harmonic: exactly satisfies
      any pure-Laplacian elliptic operator, in any number of dimensions,
      since it's independent of coordinates beyond the first two).
    - 0 spatial coordinates (only time): t^2.
    - If a time coordinate is present, multiply by exp(-t) so the
      candidate is genuinely time-dependent for transient equations too.
    """
    spatial = [c for c in coord_syms if str(c) not in _TIME_COORD_NAMES]
    time_syms = [c for c in coord_syms if str(c) in _TIME_COORD_NAMES]

    if not spatial:
        base = time_syms[0] ** 2 if time_syms else sp.Integer(1)
    elif len(spatial) == 1:
        base = sp.sin(sp.pi * spatial[0])
    else:
        base = spatial[0] ** 2 - spatial[1] ** 2

    if time_syms and spatial:
        base = base * sp.exp(-time_syms[0])
    return base


def run_manufactured_solution_check(
    sympy_pde: sp.Expr,
    coord_syms: Sequence[sp.Symbol],
    field_syms: Sequence[sp.Function],
    *,
    param_syms: Optional[Sequence[sp.Symbol]] = None,
    params: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-4,
    n_points: int = 256,
    seed: Optional[int] = 0,
) -> ManufacturedSolutionCheck:
    """Run a real method-of-manufactured-solutions self-consistency check
    against `sympy_pde`, the same technique as
    ``tests/test_manufactured_solutions.py``: compile `sympy_pde` with the
    real `pinneapple_physics.symbolic_pde` machinery, substitute a simple
    closed-form exact solution for every field (picked by
    `_select_exact_solution`, sharing the same profile across fields --
    see `ManufacturedSolutionCheck` docstring for that limitation),
    evaluate the compiled residual at real collocation points via real
    autograd derivatives, and compare its root-mean-square magnitude to
    `tolerance`.

    This function takes plain sympy/values, not a `DraftedPreset` --
    deliberately, so it can be exercised directly and deterministically in
    tests against hand-constructed known-correct/known-wrong PDEs, without
    depending on real LLM variability (see ``tests/test_preset_authoring.py``).

    Collocation points are sampled uniformly from
    ``[0.2, 1.2]`` per coordinate -- a fixed, generic interval chosen to
    avoid the origin (where terms like ``log`` or ``1/r`` would blow up)
    rather than anything domain-aware.
    """
    param_syms = list(param_syms or [])
    params = dict(params or {})

    pde = pde_from_sympy(sympy_pde, list(coord_syms), list(field_syms), param_syms or None)

    candidate = _select_exact_solution(coord_syms)
    candidate_fn = sp.lambdify(list(coord_syms), candidate, modules="torch")
    n_fields = max(len(field_syms), 1)
    exact_solution_used = {
        (f.__name__ if hasattr(f, "__name__") else str(f)): str(candidate) for f in field_syms
    }

    def _model_fn(coords: torch.Tensor) -> torch.Tensor:
        cols = [coords[:, i : i + 1] for i in range(len(coord_syms))]
        val = candidate_fn(*cols)
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, dtype=coords.dtype, device=coords.device).expand_as(coords[:, :1])
        return val.repeat(1, n_fields)

    model = _ExactFn(_model_fn)
    residual_fn = pde.to_residual_fn(model, params={k: torch.as_tensor(float(v)) for k, v in params.items()} or None)

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        raw = torch.rand(n_points, max(len(coord_syms), 1), generator=gen)
    else:
        raw = torch.rand(n_points, max(len(coord_syms), 1))
    coords = (_DOMAIN_LOW + raw * (_DOMAIN_HIGH - _DOMAIN_LOW)).requires_grad_(True)

    residual = residual_fn(coords)
    residual_rms = float(torch.sqrt(torch.mean(residual ** 2)).item())
    passed = residual_rms < tolerance

    notes = (
        f"exact solution {candidate} substituted for {n_fields} field(s) over "
        f"{n_points} collocation points in [{_DOMAIN_LOW}, {_DOMAIN_HIGH}]^{len(coord_syms)}; "
        f"RMS residual = {residual_rms:.6g} ({'<' if passed else '>='} tolerance {tolerance:.6g})."
    )

    return ManufacturedSolutionCheck(
        exact_solution_used=exact_solution_used,
        residual_at_exact_solution=residual_rms,
        tolerance=tolerance,
        passed=passed,
        n_points=n_points,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def draft_preset(
    phenomenon_description: str,
    *,
    llm_provider: str = "anthropic",
    model: Optional[str] = None,
    use_knowledge_base: bool = True,
    api_key: Optional[str] = None,
    tolerance: float = 1e-4,
    n_points: int = 256,
    conversation_store: Any = None,
) -> DraftedPreset:
    """Ask an LLM to draft a candidate governing PDE for `phenomenon_description`,
    then run a real manufactured-solution self-consistency check against it.

    Honesty about what this does and doesn't prove
    -----------------------------------------------
    When `use_knowledge_base` is True (the default), the prompt is grounded
    with real, citable equations from
    ``pinneapple_problemdesign.knowledge.physics_knowledge`` for any
    matching phenomena, so the LLM is nudged toward reusing/adapting a real
    equation instead of inventing one outright. This measurably *reduces*
    hallucination risk -- it does **not eliminate it**. A returned
    `DraftedPreset` with ``is_verified=True`` means the proposed equation
    is *self-consistent* with a simple textbook-style exact solution (see
    `run_manufactured_solution_check`); it is not proof the equation
    correctly models the real phenomenon described, and it is not
    automatically usable -- this function never registers anything into
    the real preset registry (`pinneapple_physics.pde_environment.presets`).
    A verified draft is data for a human (or a separately-built, more
    conservative research loop) to review and, if appropriate, formalize
    into a real preset file.

    Failure modes
    -------------
    Raises `PresetDraftError` (never silently substitutes a fallback
    equation) if:
      - the LLM's response isn't valid JSON, or is missing required keys;
      - the proposed PDE string doesn't parse as sympy;
      - the proposed PDE references an undeclared field, or a parameter
        with no supplied numeric value.

    Does NOT raise if parsing/compiling succeeds but the manufactured-
    solution check fails -- that is a legitimate, informative result
    (``is_verified=False``, with the real failure details in
    ``verification_result``), not an exceptional one. A failed check is
    exactly the kind of thing a research loop needs to know: "this
    approach didn't check out."
    """
    citations: List[str] = []
    hits: List[PhenomenonEntry] = []
    if use_knowledge_base:
        hits = _grounding_hits(phenomenon_description)
        citations = [f"{h.preset_module}.{h.preset_function}" for h in hits]

    grounding_text = _format_grounding(hits)
    prompt = _build_prompt(phenomenon_description, grounding_text)

    raw = call_llm(
        prompt,
        provider=llm_provider,
        model=model,
        api_key=api_key,
        system=_SYSTEM_PROMPT,
        json_mode=True,
        module="preset_authoring",
        conversation_store=conversation_store,
    )

    cleaned = _extract_first_json_object(_strip_code_fences(raw))
    try:
        payload = json.loads(cleaned)
    except Exception as e:
        raise PresetDraftError("llm_json_parse", f"LLM did not return valid JSON: {e}", raw=raw) from e

    if not isinstance(payload, dict):
        raise PresetDraftError("llm_json_parse", f"LLM JSON response is not an object: {payload!r}", raw=raw)

    coords = payload.get("coords") or []
    fields = payload.get("fields") or []
    params_payload = payload.get("params") or {}
    pde_str = payload.get("pde_residual") or ""
    bcs = payload.get("boundary_conditions") or []
    ics = payload.get("initial_conditions") or []

    missing_fields = [
        name for name, val in (("coords", coords), ("fields", fields), ("pde_residual", pde_str)) if not val
    ]
    if missing_fields:
        raise PresetDraftError(
            "missing_fields",
            f"LLM proposal is missing required field(s) {missing_fields}: {payload}",
            raw=raw,
        )

    sympy_pde, coord_syms, field_syms, param_syms, resolved_params = _parse_proposed_pde(
        pde_str, coords, fields, params_payload
    )

    check = run_manufactured_solution_check(
        sympy_pde, coord_syms, field_syms,
        param_syms=param_syms, params=resolved_params,
        tolerance=tolerance, n_points=n_points,
    )

    return DraftedPreset(
        phenomenon_description=phenomenon_description,
        proposed_equation_latex_or_sympy=pde_str,
        sympy_pde=sympy_pde,
        proposed_bcs=list(bcs),
        proposed_ics=list(ics),
        verification_result=check,
        is_verified=check.passed,
        citations_used=citations,
        coords=list(coords),
        fields=list(fields),
        params=resolved_params,
        raw_llm_response=raw,
    )
