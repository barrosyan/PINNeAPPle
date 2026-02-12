from __future__ import annotations

from typing import List
from ..schema import DesignReport, Gap


def _format_gaps(gaps: List[Gap]) -> str:
    if not gaps:
        return "_No gaps._"
    lines = []
    for g in gaps:
        status = "✅ resolved" if g.resolved else "❗ pending"
        lines.append(f"- **[{g.severity}] {g.id}** — {g.question} ({status})")
        if g.rationale:
            lines.append(f"  - Rationale: {g.rationale}")
        if g.how_to_obtain:
            lines.append(f"  - How to obtain: {g.how_to_obtain}")
    return "\n".join(lines)


def render_markdown_report(report: DesignReport, warnings: List[str]) -> str:
    s = report.spec
    p = report.plan

    md = []
    md.append("# Pinneaple Design Report\n")

    md.append("## 1) Summary\n")
    md.append(f"**Title:** {s.title or '_unset_'}\n")
    md.append(f"**Goal:** {s.goal or '_unset_'}\n")
    md.append(f"**Task type:** {s.task_type}\n")
    if s.domain_context:
        md.append(f"**Context:** {s.domain_context}\n")

    md.append("\n## 2) Specification\n")
    md.append(f"- **Inputs:** {', '.join(s.inputs) if s.inputs else '_unset_'}")
    md.append(f"- **Outputs:** {', '.join(s.outputs) if s.outputs else '_unset_'}")
    md.append(f"- **Frequency:** {s.frequency or '_unset_'}")
    md.append(f"- **Input window:** {s.input_window or '_unset_'}")
    md.append(f"- **Horizon:** {s.horizon or '_unset_'}")

    md.append("\n### Data\n")
    d = s.data
    md.append(f"- **Sources:** {', '.join(d.sources) if d.sources else '_unset_'}")
    md.append(f"- **Format:** {d.format or '_unset_'}")
    md.append(f"- **Sampling:** {d.sampling or '_unset_'}")
    md.append(f"- **Observed vars:** {', '.join(d.variables_observed) if d.variables_observed else '_unset_'}")
    md.append(f"- **Target vars:** {', '.join(d.target_variables) if d.target_variables else '_unset_'}")
    md.append(f"- **Val split:** {d.val_split_policy or '_unset_'}")

    md.append("\n### Physics/Constraints (if applicable)\n")
    ph = s.physics
    md.append(f"- **Equations:** {('; '.join(ph.governing_equations)) if ph.governing_equations else '_none provided_'}")
    md.append(f"- **Constraints:** {('; '.join(ph.constraints)) if ph.constraints else '_none provided_'}")

    md.append("\n### Validation\n")
    v = s.validation
    md.append(f"- **Metrics:** {', '.join(v.primary_metrics) if v.primary_metrics else '_unset_'}")
    md.append(f"- **Acceptance criteria:** {v.acceptance_criteria or '_unset_'}")

    if s.assumptions:
        md.append("\n## 3) Assumptions (need confirmation)\n")
        for a in s.assumptions:
            flag = "⚠️ confirm" if a.needs_confirmation else "ok"
            md.append(f"- {a.text} ({flag}, conf={a.confidence:.2f})")

    if warnings:
        md.append("\n## 4) Consistency warnings\n")
        for w in warnings:
            md.append(f"- ⚠️ {w}")

    md.append("\n## 5) Gaps (missing info)\n")
    md.append(_format_gaps(report.gaps))

    md.append("\n## 6) Recommended approach (Pinneaple)\n")
    md.append(f"**Recommendation:** {p.recommended_approach}\n")
    if p.alternatives:
        md.append("**Alternatives:**")
        for a in p.alternatives:
            md.append(f"- {a}")

    md.append("\n## 7) Step-by-step plan\n")
    for i, step in enumerate(p.steps, start=1):
        md.append(f"### {i}. {step.title}\n")
        md.append(f"**Why:** {step.why}\n")
        if step.actions:
            md.append("**Actions:**")
            for a in step.actions:
                md.append(f"- {a}")
        if step.pinneaple_modules:
            md.append("**Pinneaple modules involved:**")
            for m in step.pinneaple_modules:
                md.append(f"- `{m}`")
        if step.exit_criteria:
            md.append("**Exit criteria:**")
            for c in step.exit_criteria:
                md.append(f"- {c}")
        md.append("")

    if p.go_no_go:
        md.append("\n## 8) Go / No-Go\n")
        for g in p.go_no_go:
            md.append(f"- {g}")

    md.append(f"\n---\n_generated at {report.created_at}_\n")
    return "\n".join(md)
