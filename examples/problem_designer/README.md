## pinneaple_problemdesign examples

This folder showcases **problem elicitation + specification + planning**.

There are two main ways to use `pinneaple_problemdesign`:

1) **LLM-assisted, conversational** (Gemini / your own provider)
2) **Offline / deterministic** (you already have a spec, just want a plan + report)

### 0) Install (repo root)

```bash
pip install -e .
```

### 1) LLM-assisted (Gemini)

These examples require the optional dependency and an API key:

```bash
pip install google-generativeai
export GOOGLE_API_KEY=...   # Linux/macOS
# set GOOGLE_API_KEY=...    # Windows CMD
# $env:GOOGLE_API_KEY=...   # PowerShell
```

Run:

```bash
python examples/pinneaple_problemdesign/01_SimpleUseCase.py
python examples/pinneaple_problemdesign/02_Api.py
```

### 2) Offline / no LLM required

Generate a full report from a manually-filled `ProblemSpec`:

```bash
python examples/pinneaple_problemdesign/03_offline_spec_to_report.py
```

### 3) Plug your own provider (no vendor lock-in)

This example shows a minimal adapter that implements the internal `LLMProvider` protocol.
It uses a deterministic "mock" provider so you can run without API keys:

```bash
python examples/pinneaple_problemdesign/04_custom_provider_mock_end_to_end.py
```

### 4) Batch mode (generate multiple reports)

```bash
python examples/pinneaple_problemdesign/05_batch_generate_reports.py
```

It writes Markdown + JSON reports to:

```
artifacts/problemdesign/
```
