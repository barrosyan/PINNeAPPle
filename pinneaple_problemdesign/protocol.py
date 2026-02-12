from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import google.generativeai as genai


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(Protocol):
    """
    Single interface for the module. Plug Gemini in here.
    """
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
    """
    Real Gemini provider using Google Generative AI SDK.
    Requires environment variable: GOOGLE_API_KEY
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def _convert_messages(self, messages: List[LLMMessage]):
        """
        Convert internal message format to Gemini format.
        Gemini expects:
        [
            {"role": "user", "parts": ["text"]},
            {"role": "model", "parts": ["text"]}
        ]
        """
        converted = []
        system_prompt = None

        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role == "user":
                converted.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                converted.append({"role": "model", "parts": [m.content]})

        return system_prompt, converted

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> LLMResponse:

        system_prompt, converted_messages = self._convert_messages(messages)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        chat = self.model.start_chat(history=converted_messages[:-1])

        if system_prompt:
            prompt = f"{system_prompt}\n\n{converted_messages[-1]['parts'][0]}"
            response = chat.send_message(prompt, generation_config=generation_config)
        else:
            response = chat.send_message(
                converted_messages[-1]["parts"][0],
                generation_config=generation_config,
            )

        return LLMResponse(
            text=response.text,
            raw=response.to_dict() if hasattr(response, "to_dict") else None,
        )
