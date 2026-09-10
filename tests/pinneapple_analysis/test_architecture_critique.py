"""Tests for pinneapple_analysis.verification.architecture_critique.

Structural/rejection tests stub pinneapple_llm.call_llm (no Ollama
needed, same pattern as geometry_intelligence's LLM tests) -- exercising
the exact rejection code a genuinely hallucinated live response would
hit. One live test is gated on a real local Ollama server, matching the
rest of this codebase's convention.
"""
from __future__ import annotations

import json

import pytest

import pinneapple_llm as pl
from pinneapple_analysis.verification.architecture_critique import (
    FAILURE_MODE_CHECKLIST,
    CritiqueFinding,
    AdversarialReviewReport,
    run_adversarial_review,
)

_OLLAMA_MODEL = "llama3.2:3b"


def _ollama_reachable() -> bool:
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.0)
        have = {m.get("name", "") for m in r.json().get("models", [])}
        return _OLLAMA_MODEL in have or f"{_OLLAMA_MODEL}:latest" in have
    except Exception:
        return False


def _complete_valid_response(overrides: dict | None = None) -> str:
    """A well-formed response addressing every checklist category with
    verdict='no_issue_found' (the simplest valid shape), optionally
    overriding specific per-category entries via `overrides`
    ({category: {field: value}})."""
    overrides = overrides or {}
    findings = []
    for cat in FAILURE_MODE_CHECKLIST:
        entry = {"category": cat, "verdict": "no_issue_found", "severity": None,
                  "reasoning": "no specific concern identified"}
        entry.update(overrides.get(cat, {}))
        findings.append(entry)
    return json.dumps({"findings": findings, "overall_reasoning": "no major concerns"})


def test_valid_complete_response_parses_into_report(monkeypatch):
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response())
    report = run_adversarial_review("a plain FNO surrogate", provider="ollama", model=_OLLAMA_MODEL)

    assert isinstance(report, AdversarialReviewReport)
    assert len(report.findings) == len(FAILURE_MODE_CHECKLIST)
    assert all(isinstance(f, CritiqueFinding) for f in report.findings)
    assert {f.category for f in report.findings} == set(FAILURE_MODE_CHECKLIST)
    assert report.concerns == []
    assert report.high_severity_concerns == []


def test_response_with_a_concern_populates_severity(monkeypatch):
    overrides = {"data_leakage": {"verdict": "concern", "severity": "high",
                                   "reasoning": "normalization computed over the full dataset"}}
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response(overrides))
    report = run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)

    assert len(report.concerns) == 1
    assert report.concerns[0].category == "data_leakage"
    assert len(report.high_severity_concerns) == 1


def test_invalid_json_is_rejected(monkeypatch):
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: "not json at all")
    with pytest.raises(ValueError, match="did not return valid JSON"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_hallucinated_category_is_rejected(monkeypatch):
    def _fake(*a, **k):
        return json.dumps({
            "findings": [{"category": "made_up_category", "verdict": "no_issue_found",
                           "severity": None, "reasoning": "x"}],
            "overall_reasoning": "x",
        })
    monkeypatch.setattr(pl, "call_llm", _fake)
    with pytest.raises(ValueError, match="not in the real checklist"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_hallucinated_verdict_is_rejected(monkeypatch):
    overrides = {"data_leakage": {"verdict": "probably_fine"}}
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response(overrides))
    with pytest.raises(ValueError, match="not in the real allowed set"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_missing_category_is_rejected(monkeypatch):
    def _fake(*a, **k):
        cats = list(FAILURE_MODE_CHECKLIST)[:-1]  # drop the last category
        findings = [{"category": c, "verdict": "no_issue_found", "severity": None, "reasoning": "x"}
                    for c in cats]
        return json.dumps({"findings": findings, "overall_reasoning": "x"})
    monkeypatch.setattr(pl, "call_llm", _fake)
    with pytest.raises(ValueError, match="did not address every checklist category"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_concern_without_severity_is_rejected(monkeypatch):
    overrides = {"data_leakage": {"verdict": "concern", "severity": None}}
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response(overrides))
    with pytest.raises(ValueError, match="severity"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_non_concern_with_severity_is_rejected(monkeypatch):
    overrides = {"data_leakage": {"verdict": "not_applicable", "severity": "low"}}
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response(overrides))
    with pytest.raises(ValueError, match="severity must be null"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_hallucinated_severity_value_is_rejected(monkeypatch):
    overrides = {"data_leakage": {"verdict": "concern", "severity": "catastrophic"}}
    monkeypatch.setattr(pl, "call_llm", lambda *a, **k: _complete_valid_response(overrides))
    with pytest.raises(ValueError, match="not in"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_duplicate_category_still_requires_all_categories_present(monkeypatch):
    """A response that repeats one category and omits another must still
    fail the completeness check -- exactly one finding per real category,
    no more, no fewer, matching the system prompt's own stated rule."""
    def _fake(*a, **k):
        cats = list(FAILURE_MODE_CHECKLIST)
        findings = [{"category": cats[0], "verdict": "no_issue_found", "severity": None, "reasoning": "x"}
                    for _ in range(2)]  # duplicate the first category
        findings += [{"category": c, "verdict": "no_issue_found", "severity": None, "reasoning": "x"}
                     for c in cats[1:-1]]  # everything except the last
        return json.dumps({"findings": findings, "overall_reasoning": "x"})
    monkeypatch.setattr(pl, "call_llm", _fake)
    with pytest.raises(ValueError, match="did not address every checklist category"):
        run_adversarial_review("desc", provider="ollama", model=_OLLAMA_MODEL)


def test_spec_context_is_included_in_prompt(monkeypatch):
    captured = {}

    def _fake(prompt, **k):
        captured["prompt"] = prompt
        return _complete_valid_response()

    monkeypatch.setattr(pl, "call_llm", _fake)

    class _FakePDE:
        kind = "burgers_1d"

    class _FakeSpec:
        pde = _FakePDE()
        coords = ["x", "t"]
        domain_bounds = {"x": (0.0, 1.0), "t": (0.0, 1.0)}

    run_adversarial_review("desc", spec=_FakeSpec(), provider="ollama", model=_OLLAMA_MODEL)
    assert "burgers_1d" in captured["prompt"]


@pytest.mark.skipif(not _ollama_reachable(), reason="no local Ollama server with llama3.2:3b reachable")
def test_real_llm_produces_a_valid_complete_response():
    report = run_adversarial_review(
        "A Fourier Neural Operator trained on 200 OpenFOAM channel-flow simulations, "
        "used to predict flow fields for new inlet velocities not seen during training, "
        "with no uncertainty quantification and no OOD detection at inference time.",
        provider="ollama", model=_OLLAMA_MODEL,
    )
    assert len(report.findings) == len(FAILURE_MODE_CHECKLIST)
    assert {f.category for f in report.findings} == set(FAILURE_MODE_CHECKLIST)
    # This specific description has an obvious, real gap (no OOD detection despite
    # querying outside the training distribution) -- a genuinely useful review should
    # flag SOMETHING as a concern, not rubber-stamp it.
    assert len(report.concerns) >= 1
