from __future__ import annotations

from typing import Protocol, Optional, Dict, Any


class LLMProvider(Protocol):
    """
    Provider interface used by pinneaple_problemdesign.

    Contract:
      - generate(...) returns a string (usually YAML/JSON)
      - __call__(...) is an alias to generate(...)
    """

    def generate(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...

    def __call__(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...


# Local providers shipped in this repo
from .cosmos_provider import CosmosProvider  # noqa: E402

__all__ = [
    "LLMProvider",
    "CosmosProvider",
]
