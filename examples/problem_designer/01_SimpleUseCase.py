import os

from pinneaple_problemdesign import GeminiProvider, DesignAgent


def _build_agent() -> DesignAgent:
    """Create agent with good error messages for optional deps/env."""
    try:
        llm = GeminiProvider(model="gemini-2.0-flash")
    except ModuleNotFoundError as e:
        raise SystemExit(
            "\n".join(
                [
                    "[ERROR] This example requires the Google Gemini SDK.",
                    "Install:",
                    "  pip install google-generativeai",
                    "",
                    "Then set your API key:",
                    "  export GOOGLE_API_KEY=...   (Linux/macOS)",
                    "  set GOOGLE_API_KEY=...      (Windows CMD)",
                    "  $env:GOOGLE_API_KEY=...     (PowerShell)",
                ]
            )
        ) from e
    except ValueError as e:
        raise SystemExit(
            "\n".join(
                [
                    "[ERROR] Missing GOOGLE_API_KEY.",
                    "Set it before running:",
                    "  export GOOGLE_API_KEY=...   (Linux/macOS)",
                    "  set GOOGLE_API_KEY=...      (Windows CMD)",
                    "  $env:GOOGLE_API_KEY=...     (PowerShell)",
                ]
            )
        ) from e

    return DesignAgent(llm=llm)


agent = _build_agent()
state = agent.start()

user_messages = [
    "I want to forecast outlet temperature for a heat exchanger.",
    "Sampling is every 10 minutes. I need a 6-hour horizon. Inputs are inlet temp, flow rate, and ambient temp. Output is outlet temp.",
    "Data is in parquet files, 2 years. There are some missing points during maintenance. Use last 3 months for validation.",
    "Primary metric should be MAE. Success is MAE < 0.5°C at 6 hours.",
]

for msg in user_messages:
    agent.ingest_user_message(state, msg)
    out = agent.step(state)

    if out["type"] == "questions":
        print("\n--- QUESTIONS ---")
        print(out["questions_text"])
    else:
        print("\n--- FINAL REPORT ---")
        print(out["markdown"])
        break

if not state.done:
    print("\nStill pending gaps:", len(state.unresolved_gaps()))