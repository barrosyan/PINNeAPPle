"""Provider-agnostic LLM call dispatch, shared by ``draft.py``,
``geometry_draft.py``, ``research.py`` and ``twin_draft.py`` -- one place
that knows how to reach Anthropic, OpenAI, or a local Ollama server, and
one place that logs every call to a :class:`ConversationStore` when one is
configured (see that module's docstring for why: auditability of what was
actually asked/answered, and the dataset :mod:`finetune` trains on).

Logging is opt-in per call (pass ``conversation_store=``) rather than a
hidden global default, so "every LLM call in this process gets written to
disk" is something a caller chooses, not something that happens to them.
"""
from __future__ import annotations

from typing import Optional

_PROVIDERS = ("anthropic", "openai", "ollama")


def call_llm(
    prompt: str,
    *,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system: str = "",
    json_mode: bool = False,
    module: str = "unknown",
    conversation_store=None,
    **provider_kwargs,
) -> str:
    """Dispatch a single prompt to whichever provider is requested and
    return the raw text response. See individual provider modules for
    what ``**provider_kwargs`` accepts (``host``/``port``/``timeout`` for
    ``"ollama"``; unused for the others).
    """
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or "claude-sonnet-5",
            max_tokens=1024,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        response = "".join(block.text for block in msg.content if hasattr(block, "text"))

    elif provider == "openai":
        import openai

        client = openai.OpenAI(api_key=api_key)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=model or "gpt-4o",
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        response = resp.choices[0].message.content

    elif provider == "ollama":
        from .local_llm import call_ollama

        response = call_ollama(prompt, model or "llama3.1", api_key, system, **provider_kwargs)

    else:
        raise ValueError(f"unknown provider '{provider}', expected one of {_PROVIDERS}")

    if conversation_store is not None:
        conversation_store.log(
            module=module, provider=provider, model=model,
            system_prompt=system or None, user_prompt=prompt, response=response,
        )

    return response
