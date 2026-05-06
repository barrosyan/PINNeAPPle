"""Offline usage: build a full DesignReport without any LLM.

Why this matters:
  - You can integrate `pinneaple_problemdesign` into CI / pipelines.
  - You can generate consistent reports for stakeholders.
  - You can start from a partially-known spec and still get a structured plan.

Run:
  python examples/pinneaple_problemdesign/03_offline_spec_to_report.py
"""

from __future__ import annotations

from pathlib import Path

from pinneaple_problemdesign.schema import (
    ProblemSpec,
    DataSpec,
    ValidationSpec,
    DesignReport,
)
from pinneaple_problemdesign.knowledge.mapping import build_plan_fno_first
from pinneaple_problemdesign.renderers.report_md import render_markdown_report
from pinneaple_problemdesign.renderers.report_json import render_json_report
from pinneaple_problemdesign.elicitation.validators import validate_and_suggest


def main() -> None:
    # 1) Fill what you already know (this can come from a form, config, DB, etc.)
    spec = ProblemSpec(
        title="Heat Exchanger: Outlet Temperature Forecast",
        goal="Forecast outlet temperature 6 hours ahead to optimize energy usage",
        task_type="forecasting",
        inputs=["T_in (°C)", "flow_rate (kg/s)", "T_ambient (°C)", "pressure (bar)"],
        outputs=["T_out (°C)"],
        frequency="10min",
        input_window="48 steps (8h)",
        horizon="6h",
        domain_context=(
            "Industrial heat exchanger. Maintenance causes intermittent missing data. "
            "We expect seasonal drift across weeks/months."
        ),
        data=DataSpec(
            sources=["s3://plant-data/hex/parquet/"],
            format="parquet",
            sampling="10min",
            variables_observed=["T_in", "flow_rate", "T_ambient", "pressure", "T_out"],
            target_variables=["T_out"],
            known_quality_issues=["missing points during maintenance", "sensor spikes"],
            missingness="bursty gaps during maintenance windows",
            train_span="2023-01-01 .. 2025-01-01",
            val_split_policy="last 3 months as validation (time-based)",
            labels_available=True,
        ),
        validation=ValidationSpec(
            primary_metrics=["MAE"],
            acceptance_criteria="MAE < 0.5°C at 6h horizon",
            robustness_tests=["missingness stress", "drift across weeks", "extreme ambient temp"],
        ),
    )

    # 2) Run validators (warnings + safe assumptions)
    warnings, assumptions = validate_and_suggest(spec)
    spec.assumptions.extend(assumptions)

    # 3) Build plan and report
    gaps = []  # if you already know gaps, add them here; otherwise keep empty.
    plan = build_plan_fno_first(spec, gaps)
    report = DesignReport(spec=spec, gaps=gaps, plan=plan)

    md = render_markdown_report(report, warnings)
    js = render_json_report(report)

    # 4) Save artifacts
    out_dir = Path("artifacts/problemdesign")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline_report.md").write_text(md, encoding="utf-8")
    (out_dir / "offline_report.json").write_text(js, encoding="utf-8")

    print("Wrote:")
    print(" -", out_dir / "offline_report.md")
    print(" -", out_dir / "offline_report.json")


if __name__ == "__main__":
    main()
