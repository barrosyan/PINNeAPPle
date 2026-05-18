"""LLMProvider protocol, LLMMessage, LLMResponse, GeminiProvider.

This module intentionally treats external LLM SDKs as **optional dependencies**.

- If `google-generativeai` is not installed, importing this module will still work,
  but instantiating `GeminiProvider` will raise a clear, actionable error.

Install:
  pip install google-generativeai
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(Protocol):
    """Single interface for the module."""

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...


class GeminiProvider:
    """Gemini provider using Google Generative AI SDK.

    Requires:
      - `google-generativeai` package
      - environment variable: `GOOGLE_API_KEY`
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        if genai is None:
            raise ModuleNotFoundError(
                "google-generativeai is not installed. Install it with: pip install google-generativeai"
            )

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:
        # Convert to Gemini chat format
        parts = []
        for m in messages:
            parts.append({"role": m.role, "parts": [m.content]})

        # JSON mode is best-effort; Gemini SDK behavior may vary by model.
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        resp = self.model.generate_content(parts, generation_config=generation_config)
        text = getattr(resp, "text", None)
        if text is None and hasattr(resp, "candidates") and resp.candidates:
            # fallback
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        return LLMResponse(text=text or "", raw={"_provider": "gemini"})