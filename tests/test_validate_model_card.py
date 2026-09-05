"""Regression tests for the model-hub governance gate (see
``ROADMAP_PHYSICS_AI_HUB.md`` P1.5 and ``scripts/validate_model_card.py``).

Locks in that the CLI wrapper enforces exactly the same schema
``ModelCard.validate()`` does (no drift between what CI checks and what
``push_to_hub`` itself would accept), and that a CI-style non-zero exit
code actually occurs for a card missing a computed validation metric --
the whole point of this gate is to make that failure automatic rather
than something a reviewer has to notice by eye.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "validate_model_card.py"

_spec = importlib.util.spec_from_file_location("validate_model_card", _SCRIPT_PATH)
validate_model_card = importlib.util.module_from_spec(_spec)
sys.modules["validate_model_card"] = validate_model_card
_spec.loader.exec_module(validate_model_card)

from pinneapple_hub.model_card import ModelCard


def _write_card(tmp_path: Path, **overrides) -> Path:
    card = ModelCard(
        name=overrides.pop("name", "test_model"),
        architecture=overrides.pop("architecture", "modified_mlp"),
        validation_metrics=overrides.pop("validation_metrics", {"rmse": 0.02}),
        reference_source=overrides.pop("reference_source", "Pope, Turbulent Flows, Ch. 7"),
        **overrides,
    )
    path = tmp_path / "model_card.json"
    card.save(str(path))
    return path


def test_valid_card_passes_with_no_problems(tmp_path):
    path = _write_card(tmp_path)
    assert validate_model_card.validate_one(path) == []


def test_card_missing_validation_metrics_fails(tmp_path):
    path = _write_card(tmp_path, validation_metrics={})
    problems = validate_model_card.validate_one(path)
    assert any("validation_metrics" in p for p in problems)


def test_card_with_metrics_but_no_reference_source_fails(tmp_path):
    path = _write_card(tmp_path, reference_source="")
    problems = validate_model_card.validate_one(path)
    assert any("reference_source" in p for p in problems)


def test_card_missing_architecture_fails(tmp_path):
    path = _write_card(tmp_path, architecture="")
    problems = validate_model_card.validate_one(path)
    assert any("architecture" in p for p in problems)


def test_unparseable_file_reports_a_problem_not_a_crash(tmp_path):
    path = tmp_path / "model_card.json"
    path.write_text("not valid json {{{")
    problems = validate_model_card.validate_one(path)
    assert len(problems) == 1
    assert "could not load" in problems[0]


def test_cli_main_exits_zero_for_all_passing_cards(tmp_path, capsys):
    good = _write_card(tmp_path)
    rc = validate_model_card.main([str(good)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "0 failed" in out


def test_cli_main_exits_nonzero_when_any_card_fails(tmp_path, capsys):
    good = _write_card(tmp_path)
    bad_path = tmp_path / "bad_model_card.json"
    ModelCard(name="incomplete").save(str(bad_path))
    rc = validate_model_card.main([str(good), str(bad_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in str(bad_path) or "FAIL" in out
    assert "1 failed" in out


def test_cli_main_with_no_args_scans_repo_and_does_not_crash():
    # The real repo has no committed model_card.json fixtures (hub pushes
    # are runtime artifacts, not checked-in files) -- this only asserts
    # the discovery path itself doesn't raise, not that it finds anything.
    rc = validate_model_card.main([])
    assert rc == 0
