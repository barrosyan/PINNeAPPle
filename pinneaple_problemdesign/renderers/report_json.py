from __future__ import annotations

import json
from ..schema import DesignReport


def render_json_report(report: DesignReport, *, indent: int = 2) -> str:
    payload = {
        "created_at": report.created_at,
        "spec": report.spec.to_dict(),
        "gaps": [g.__dict__ for g in report.gaps],
        "plan": {
            "recommended_approach": report.plan.recommended_approach,
            "alternatives": report.plan.alternatives,
            "steps": [
                {
                    "title": s.title,
                    "why": s.why,
                    "actions": s.actions,
                    "pinneaple_modules": s.pinneaple_modules,
                    "exit_criteria": s.exit_criteria,
                }
                for s in report.plan.steps
            ],
            "go_no_go": report.plan.go_no_go,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=indent)
