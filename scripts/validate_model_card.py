#!/usr/bin/env python3
"""Governance-as-code gate for the model hub (see ``ROADMAP_PHYSICS_AI_HUB.md``,
P1.5): reject a model-card push with no computed ``validation_metrics`` or no
``reference_source`` citation, the same schema check ``ModelCard.validate()``
already implements -- this script is just the CI-callable/CLI wrapper around
it, so the rule lives in one place (``pinneapple_hub/model_card.py``) and
this script can't drift from what ``push_to_hub`` itself enforces.

Usage
-----
    python scripts/validate_model_card.py path/to/model_card.json [more.json ...]

With no arguments, scans the repo for every ``*model_card*.json`` file (the
naming convention ``pinneapple_hub`` writes, e.g. ``model_card.json`` next to
a pushed checkpoint) and validates each one found. Exits non-zero if any
file fails schema validation or is not valid JSON for a ``ModelCard`` at all,
so it can be wired into CI as a required check (see
``.github/workflows/tests.yml``) without needing a hand-maintained file
list -- a PR that adds or edits a model card is caught automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pinneapple_hub.model_card import ModelCard  # noqa: E402


def _discover_model_cards() -> List[Path]:
    return sorted(
        p for p in _REPO_ROOT.rglob("*model_card*.json")
        if ".git" not in p.parts and "node_modules" not in p.parts
    )


def validate_one(path: Path) -> List[str]:
    """Return a list of problems (empty = passes). Never raises for a
    schema issue -- only for a file that isn't readable/parseable JSON at
    all, which is itself reported as a single problem rather than a stack
    trace, so a CI log stays readable."""
    try:
        card = ModelCard.load(str(path))
    except Exception as e:
        return [f"could not load as a ModelCard: {e}"]
    return card.validate()


def main(argv: List[str]) -> int:
    paths = [Path(a) for a in argv] if argv else _discover_model_cards()
    if not paths:
        print("validate_model_card: no model_card.json files found -- nothing to check.")
        return 0

    failed = 0
    for path in paths:
        problems = validate_one(path)
        if problems:
            failed += 1
            print(f"FAIL {path}")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"PASS {path}")

    print(f"\n{len(paths)} model card(s) checked, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
