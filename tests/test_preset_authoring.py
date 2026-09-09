"""Tests for ``pinneapple_problemdesign.preset_authoring`` -- the LLM-
proposes/code-verifies preset-drafting pipeline.

Two tiers, same reasoning as ``tests/test_llm_agent_loop.py`` /
``tests/test_llm_cad_generation.py``:

1. The safety-critical logic -- `run_manufactured_solution_check` and
   `draft_preset`'s hard-failure paths -- is tested directly and
   deterministically, without any real LLM call (a hand-built known-
   correct PDE must pass, a hand-built known-wrong one must fail; a
   mocked LLM response that can't be parsed must raise
   `PresetDraftError`, not silently substitute anything).
2. If a real local Ollama server is reachable, one real end-to-end
   `draft_preset(...)` call is also run against it, for a simple, well-
   known phenomenon. Either outcome (the LLM's proposal passes its own
   generated verification, fails it, or is rejected outright as
   unparseable) is a valid, informative result -- the point is that the
   safety mechanism ran for real against a real LLM call.
"""
from __future__ import annotations

import json

import pytest
import sympy as sp

from pinneapple_problemdesign.preset_authoring import (
    DraftedPreset,
    ManufacturedSolutionCheck,
    PresetDraftError,
    draft_preset,
    run_manufactured_solution_check,
)
from pinneapple_problemdesign import preset_authoring as _pa_module


# ---------------------------------------------------------------------------
# 1a. Manufactured-solution check, direct + deterministic
# ---------------------------------------------------------------------------

def test_manufactured_check_passes_for_known_correct_1d_poisson():
    """u_xx + pi^2*sin(pi*x) = 0 is EXACTLY solved by u = sin(pi*x)
    (u_xx = -pi^2*sin(pi*x)), which is precisely the candidate
    `_select_exact_solution` picks for a single spatial coordinate -- so
    this known-correct PDE must pass with a tiny residual."""
    x = sp.Symbol("x")
    u = sp.Function("u")
    expr = sp.Derivative(u(x), x, 2) + sp.pi ** 2 * sp.sin(sp.pi * x)

    check = run_manufactured_solution_check(expr, [x], [u], tolerance=1e-3)

    assert isinstance(check, ManufacturedSolutionCheck)
    assert check.passed is True
    assert check.residual_at_exact_solution < 1e-3
    assert check.tolerance == 1e-3
    assert "u" in check.exact_solution_used


def test_manufactured_check_fails_for_known_wrong_1d_poisson():
    """u_xx + 5 = 0 has a constant, coordinate-independent source term
    that sin(pi*x)'s second derivative (-pi^2*sin(pi*x)) cannot cancel
    for any x -- the residual must be clearly, measurably nonzero. This
    is the other half of the check: a check that reports "near zero" for
    both the correct and the broken equation would be exactly as useless
    as a residual implementation that always returns ~0 regardless of
    input (the same reasoning ``test_manufactured_solutions.py`` uses for
    its own wrong-solution tests)."""
    x = sp.Symbol("x")
    u = sp.Function("u")
    expr = sp.Derivative(u(x), x, 2) + sp.Float(5.0)

    check = run_manufactured_solution_check(expr, [x], [u], tolerance=1e-3)

    assert check.passed is False
    assert check.residual_at_exact_solution > 1.0, (
        f"expected a clearly nonzero residual for the broken PDE, got {check.residual_at_exact_solution}"
    )


def test_manufactured_check_passes_for_known_correct_2d_laplace():
    """u_xx + u_yy = 0 is exactly satisfied by u = x^2 - y^2 (harmonic),
    which is the candidate `_select_exact_solution` picks for >=2 spatial
    coordinates -- reuses the same exact solution as
    ``test_manufactured_solutions.py``'s own Laplace-2D check."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    expr = sp.Derivative(u(x, y), x, 2) + sp.Derivative(u(x, y), y, 2)

    check = run_manufactured_solution_check(expr, [x, y], [u], tolerance=1e-4)

    assert check.passed is True
    assert check.residual_at_exact_solution < 1e-4


def test_manufactured_check_fails_for_known_wrong_2d_laplace_like():
    """u_xx + u_yy - 4 = 0 (Poisson with a nonzero constant source) is
    NOT satisfied by the harmonic x^2 - y^2 candidate (whose Laplacian is
    identically 0, not 4) -- residual must be clearly nonzero."""
    x, y = sp.symbols("x y")
    u = sp.Function("u")
    expr = sp.Derivative(u(x, y), x, 2) + sp.Derivative(u(x, y), y, 2) - sp.Float(4.0)

    check = run_manufactured_solution_check(expr, [x, y], [u], tolerance=1e-4)

    assert check.passed is False
    assert check.residual_at_exact_solution > 1.0


def test_manufactured_check_respects_tolerance_boundary():
    """A slightly-off but small residual should flip pass/fail depending
    purely on the requested tolerance -- confirms `passed` is actually
    derived from comparing to `tolerance`, not hardcoded."""
    x = sp.Symbol("x")
    u = sp.Function("u")
    expr = sp.Derivative(u(x), x, 2) + sp.pi ** 2 * sp.sin(sp.pi * x)

    loose = run_manufactured_solution_check(expr, [x], [u], tolerance=1.0)
    tight = run_manufactured_solution_check(expr, [x], [u], tolerance=1e-12)

    assert loose.passed is True
    assert tight.passed is False
    assert loose.residual_at_exact_solution == pytest.approx(tight.residual_at_exact_solution, rel=1e-3)


def test_manufactured_check_with_a_free_parameter():
    """D*u_xx + D*pi^2*sin(pi*x) = 0 -- exact for ANY D (D factors out),
    exercising the `param_syms`/`params` plumbing through
    `pde_from_sympy`/`to_residual_fn`."""
    x, D = sp.symbols("x D")
    u = sp.Function("u")
    expr = D * sp.Derivative(u(x), x, 2) + D * sp.pi ** 2 * sp.sin(sp.pi * x)

    check = run_manufactured_solution_check(expr, [x], [u], param_syms=[D], params={"D": 2.5}, tolerance=1e-3)
    assert check.passed is True


# ---------------------------------------------------------------------------
# 1b. draft_preset's hard-failure paths (mocked LLM, deterministic)
# ---------------------------------------------------------------------------

def _mock_call_llm(monkeypatch, response_text: str) -> None:
    def _fake(prompt, *, provider="anthropic", model=None, api_key=None, system="", json_mode=False,
              module="unknown", conversation_store=None, **kwargs):
        return response_text

    monkeypatch.setattr(_pa_module, "call_llm", _fake)


def test_draft_preset_raises_on_unparseable_json(monkeypatch):
    _mock_call_llm(monkeypatch, "this is not json at all")
    with pytest.raises(PresetDraftError) as excinfo:
        draft_preset("some phenomenon", use_knowledge_base=False)
    assert excinfo.value.stage == "llm_json_parse"


def test_draft_preset_raises_on_missing_required_fields(monkeypatch):
    _mock_call_llm(monkeypatch, json.dumps({"coords": ["x"], "fields": [], "pde_residual": ""}))
    with pytest.raises(PresetDraftError) as excinfo:
        draft_preset("some phenomenon", use_knowledge_base=False)
    assert excinfo.value.stage == "missing_fields"


def test_draft_preset_raises_on_unparseable_pde_string(monkeypatch):
    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "this ][ is not >> valid sympy (((",
        "boundary_conditions": [], "initial_conditions": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))
    with pytest.raises(PresetDraftError) as excinfo:
        draft_preset("some phenomenon", use_knowledge_base=False)
    assert excinfo.value.stage == "pde_parse"


def test_draft_preset_raises_on_undeclared_field(monkeypatch):
    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "Derivative(u(x), x, 2) + v(x)",
        "boundary_conditions": [], "initial_conditions": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))
    with pytest.raises(PresetDraftError) as excinfo:
        draft_preset("some phenomenon", use_knowledge_base=False)
    assert excinfo.value.stage == "undeclared_field"


def test_draft_preset_raises_on_missing_param_value(monkeypatch):
    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "Derivative(u(x), x, 2) + k*sin(pi*x)",
        "boundary_conditions": [], "initial_conditions": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))
    with pytest.raises(PresetDraftError) as excinfo:
        draft_preset("some phenomenon", use_knowledge_base=False)
    assert excinfo.value.stage == "missing_param_value"


def test_draft_preset_never_raises_on_a_failed_but_parseable_proposal(monkeypatch):
    """A proposal that parses and compiles fine but does NOT self-
    consistently admit the picked exact solution must come back as a
    normal `DraftedPreset` with `is_verified=False` -- NOT an exception.
    A failed check is useful information, not an error."""
    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "Derivative(u(x), x, 2) + 5.0",
        "boundary_conditions": ["u(0)=0"], "initial_conditions": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))

    result = draft_preset("a made-up broken phenomenon", use_knowledge_base=False, tolerance=1e-3)

    assert isinstance(result, DraftedPreset)
    assert result.is_verified is False
    assert result.verification_result.passed is False
    assert isinstance(result.sympy_pde, sp.Basic)


def test_draft_preset_returns_verified_true_for_a_self_consistent_mocked_proposal(monkeypatch):
    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "Derivative(u(x), x, 2) + pi**2*sin(pi*x)",
        "boundary_conditions": ["u(0)=0", "u(1)=0"], "initial_conditions": [],
        "grounded_in": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))

    result = draft_preset("1D steady conduction with a sinusoidal source", use_knowledge_base=False, tolerance=1e-3)

    assert result.is_verified is True
    assert result.verification_result.passed is True
    assert result.coords == ["x"]
    assert result.fields == ["u"]
    assert result.proposed_equation_latex_or_sympy == payload["pde_residual"]


def test_draft_preset_never_registers_into_the_real_preset_registry(monkeypatch):
    """This module must never silently promote a verified draft into the
    real, trusted preset catalog -- confirm the real registry is
    untouched by a `draft_preset` call, verified or not."""
    from pinneapple_physics.pde_environment.presets.registry import list_presets

    before = list_presets()

    payload = {
        "coords": ["x"], "fields": ["u"], "params": {},
        "pde_residual": "Derivative(u(x), x, 2) + pi**2*sin(pi*x)",
        "boundary_conditions": [], "initial_conditions": [],
    }
    _mock_call_llm(monkeypatch, json.dumps(payload))
    result = draft_preset("some phenomenon", use_knowledge_base=False)

    assert result.is_verified is True  # even a verified draft must not be auto-registered
    assert list_presets() == before


# ---------------------------------------------------------------------------
# 2. Real local-model (Ollama) end-to-end test
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
def test_real_ollama_draft_preset_end_to_end_reports_a_real_verification_outcome():
    """Real LLM, real JSON parse, real sympy parse, real compiled
    residual, real autograd -- no mocking anywhere in this test. A small
    local model is not 100% reliable at following the coords/fields/
    Derivative(...) DSL on every attempt (same caveat
    ``test_llm_cad_generation.py`` documents for its own real-model
    tests), so retry a few times; a `PresetDraftError` on every attempt
    (the LLM never produced a parseable proposal) is itself a real,
    reportable outcome of the safety mechanism working as designed --
    NOT swallowed as a test failure."""
    description = "1D heat conduction with a sinusoidal heat source"

    last_result = None
    last_error = None
    for _attempt in range(5):
        try:
            last_result = draft_preset(
                description, llm_provider="ollama", model=_OLLAMA_MODEL,
                use_knowledge_base=True, tolerance=1e-2,
            )
            last_error = None
            break
        except PresetDraftError as e:
            last_error = e
            continue

    if last_result is None:
        # Every attempt was rejected outright by the hard-failure path --
        # a valid, informative outcome: the safety mechanism caught an
        # unparseable/inconsistent proposal from a real LLM every time.
        print(f"\n[real-Ollama] draft_preset never produced a checkable proposal in 5 attempts; "
              f"last PresetDraftError: stage={last_error.stage!r} message={last_error.message!r}")
        assert last_error is not None
        return

    assert isinstance(last_result, DraftedPreset)
    assert isinstance(last_result.sympy_pde, sp.Basic)
    assert isinstance(last_result.verification_result, ManufacturedSolutionCheck)
    print(
        f"\n[real-Ollama] draft_preset('{description}') -> "
        f"proposed: {last_result.proposed_equation_latex_or_sympy!r} | "
        f"is_verified={last_result.is_verified} | "
        f"residual={last_result.verification_result.residual_at_exact_solution:.6g} "
        f"(tolerance={last_result.verification_result.tolerance:.6g}) | "
        f"citations_used={last_result.citations_used}"
    )
    # Either outcome (verified or not) is valid -- the assertion here is
    # only that the pipeline ran for real end-to-end and produced a
    # coherent, internally-consistent result object.
    assert last_result.is_verified == last_result.verification_result.passed
