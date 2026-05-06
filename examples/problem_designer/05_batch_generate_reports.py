"""Batch generation of problem-design reports.

This script turns a list of "user stories" into:
  - artifacts/problemdesign/<slug>.md
  - artifacts/problemdesign/<slug>.json

It is useful for:
  - productizing a "problem intake" flow
  - generating consistent docs for clients
  - benchmarking / regression-testing your extraction prompts

By default it tries Gemini (if installed + GOOGLE_API_KEY is set).
If not available, it falls back to a deterministic mock provider.

Run:
  python examples/pinneaple_problemdesign/05_batch_generate_reports.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from pinneaple_problemdesign import DesignAgent


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60] or "report"


def _build_agent() -> DesignAgent:
    # Try Gemini first.
    try:
        from pinneaple_problemdesign import GeminiProvider

        llm = GeminiProvider(model="gemini-2.0-flash")
        return DesignAgent(llm=llm)
    except Exception:
        # Fallback to a tiny deterministic provider (no API keys required).
        import json
        from dataclasses import dataclass
        from typing import List

        from pinneaple_problemdesign.protocol import LLMProvider, LLMMessage, LLMResponse

        @dataclass
        class _FallbackMock(LLMProvider):
            def generate(
                self,
                messages: List[LLMMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int = 800,
                json_mode: bool = False,
            ) -> LLMResponse:
                if json_mode:
                    return LLMResponse(
                        text=json.dumps(
                            {
                                "partial_spec": {},
                                "unknown_fields": [
                                    "inputs",
                                    "outputs",
                                    "horizon",
                                    "validation.primary_metrics",
                                ],
                                "assumptions_suggested": [
                                    "Assume input_window = 64 steps for baseline."
                                ],
                                "gaps_suggested": [],
                            }
                        )
                    )
                return LLMResponse(text="")

        return DesignAgent(llm=_FallbackMock())


def main() -> None:
    agent = _build_agent()

    problems: List[List[str]] = [
        [
            "We need to forecast outlet temperature for a heat exchanger.",
            "Sampling is every 10 minutes. Horizon is 6 hours. Inputs: inlet temp, flow rate, ambient temp. Output: outlet temp.",
            "Data is 2 years in parquet. Missing points during maintenance. Use last 3 months for validation.",
            "Primary metric MAE. Success MAE < 0.5°C at 6h.",
        ],
        [
            "We want to detect anomalies in a compressor using multivariate sensor data.",
            "Sampling is 1Hz. Inputs are 20 sensors (pressure, vibration, temperatures).",
            "We have labels for only 200 anomalies in the last year.",
            "We want low false positives; metric should include precision/recall.",
        ],
        [
            "We want to learn a neural operator surrogate for 2D flow around an obstacle.",
            "Inputs are boundary conditions and inlet velocity; output is velocity/pressure field.",
            "We have CFD simulations on a regular grid, 2048 samples.",
            "We care about L2 error and conservation violations.",
        ],
    ]

    out_dir = Path("artifacts/problemdesign")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, convo in enumerate(problems, start=1):
        state = agent.start()
        title = f"problem-{i}"
        for msg in convo:
            agent.ingest_user_message(state, msg)
            out = agent.step(state)
            if out["type"] == "report":
                title = out["report"].spec.title or title
                md = out["markdown"]
                # JSON renderer is available via report object
                from pinneaple_problemdesign.renderers.report_json import render_json_report

                js = render_json_report(out["report"])
                break

        slug = _slug(title)
        (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")
        (out_dir / f"{slug}.json").write_text(js, encoding="utf-8")
        print(f"[{i}] wrote {slug}.md / {slug}.json")


if __name__ == "__main__":
    main()
