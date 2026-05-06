"""End-to-end demo with a custom (mock) LLMProvider.

This shows how to:
  - Avoid vendor lock-in (Gemini/OpenAI/local gateway/etc)
  - Run the full DesignAgent flow in unit tests / CI
  - Customize behavior without changing pinneaple_problemdesign internals

Run:
  python examples/pinneaple_problemdesign/04_custom_provider_mock_end_to_end.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from pinneaple_problemdesign import DesignAgent
from pinneaple_problemdesign.protocol import LLMProvider, LLMMessage, LLMResponse


@dataclass
class MockProvider(LLMProvider):
    """A deterministic provider that returns JSON extraction based on simple regex.

    Notes:
      - This is NOT meant to be smart; it is meant to be stable.
      - Replace this with your own provider (OpenAI, local LLM, Cosmos gateway, etc.).
    """

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        # The extractor sends the user's text inside a template.
        user_blob = "\n".join(m.content for m in messages if m.role == "user")
        m = re.search(r"User message:\s*(.*)", user_blob, flags=re.DOTALL)
        user_text = (m.group(1).strip() if m else user_blob).strip()

        # 1) If this is the extractor step (json_mode=True), return a structured payload.
        if json_mode:
            partial = {}
            unknown = []

            # Very small heuristics just to demonstrate the pipeline.
            if "forecast" in user_text.lower() or "prever" in user_text.lower():
                partial["task_type"] = "forecasting"

            # Sampling frequency
            freq = re.search(r"(\d+)\s*(min|minute|minutes|h|hour|hours)", user_text.lower())
            if freq and "sampling" in user_text.lower():
                partial["frequency"] = f"{freq.group(1)}{freq.group(2)}"

            # Horizon
            if "horizon" in user_text.lower() or "ahead" in user_text.lower():
                h = re.search(r"(\d+)\s*(h|hour|hours)", user_text.lower())
                if h:
                    partial["horizon"] = f"{h.group(1)}h"

            # Inputs / outputs
            if "inputs" in user_text.lower():
                # crude list after "inputs are"
                after = user_text.split("Inputs", 1)[-1]
                after = after.split("Output", 1)[0]
                items = re.split(r"[,;]", after)
                xs = [i.strip() for i in items if i.strip() and len(i.strip()) < 60]
                if xs:
                    partial["inputs"] = xs
            if "output" in user_text.lower():
                after = user_text.split("Output", 1)[-1]
                items = re.split(r"[,;.]", after)
                ys = [i.strip() for i in items if i.strip() and len(i.strip()) < 60]
                if ys:
                    partial["outputs"] = [ys[0]]

            # Data format
            if "parquet" in user_text.lower():
                partial.setdefault("data", {})
                partial["data"]["format"] = "parquet"

            # Validation metric
            if "mae" in user_text.lower():
                partial.setdefault("validation", {})
                partial["validation"]["primary_metrics"] = ["MAE"]

            # Collect unknowns we care about
            if "inputs" not in partial:
                unknown.append("inputs")
            if "outputs" not in partial:
                unknown.append("outputs")
            if "horizon" not in partial:
                unknown.append("horizon")
            if "validation" not in partial:
                unknown.append("validation.primary_metrics")

            payload = {
                "partial_spec": partial,
                "unknown_fields": unknown,
                "assumptions_suggested": ["Assume input_window = 64 steps for baseline."],
                "gaps_suggested": [],
            }
            return LLMResponse(text=json.dumps(payload))

        # 2) Otherwise, this is the optional "rewrite questions" step.
        # Keep it simple (numbered list).
        lines = []
        for i, m in enumerate(messages, start=1):
            if m.role == "user" and "Questions to rewrite" in m.content:
                qs = [ln.strip("- ") for ln in m.content.splitlines() if ln.strip().startswith("-")]
                for j, q in enumerate(qs, start=1):
                    # remove severity tag
                    q = re.sub(r"^\[[^\]]+\]\s*", "", q)
                    lines.append(f"{j}. {q}")
        return LLMResponse(text="\n".join(lines) if lines else "No questions.")


def main() -> None:
    agent = DesignAgent(llm=MockProvider())
    state = agent.start()

    # Simulate a short conversation.
    user_messages = [
        "I want to forecast outlet temperature for a heat exchanger.",
        "Sampling is every 10 minutes. I need a 6-hour horizon. Inputs are inlet temp, flow rate, and ambient temp. Output is outlet temp.",
        "Data is in parquet files, 2 years. Primary metric should be MAE.",
        "Success is MAE < 0.5°C at 6 hours.",
    ]

    for msg in user_messages:
        agent.ingest_user_message(state, msg)
        out = agent.step(state)

        if out["type"] == "questions":
            print("\n--- QUESTIONS ---")
            print(out["questions_text"])
        else:
            print("\n--- FINAL REPORT ---")
            print(out["markdown"][:1500])
            print("\n[truncated]\n")
            break

    if not state.done:
        print("\nStill pending gaps:", len(state.unresolved_gaps()))


if __name__ == "__main__":
    main()
