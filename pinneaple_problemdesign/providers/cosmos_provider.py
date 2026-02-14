from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class CosmosProviderConfig:
    """
    Cosmos provider configuration.

    This provider assumes you have *some* Cosmos gateway endpoint that can be called via HTTP.
    NVIDIA Cosmos itself is a platform of foundation models; access is typically via NVIDIA
    services (enterprise gateways/NIM).

    Environment variables supported:
      - COSMOS_BASE_URL: base URL for your gateway, e.g. http://localhost:9000
      - COSMOS_API_KEY: optional API key/bearer token for your gateway
      - COSMOS_TIMEOUT_S: request timeout seconds (default 120)
      - COSMOS_ROUTE: endpoint route path (default /scene_to_spec)

    Expected contract (request -> response):
      POST {COSMOS_BASE_URL}{COSMOS_ROUTE}
      Request JSON:
        {
          "prompt": "...",
          "system": "...",
          "schema": "...",
          "temperature": 0.2,
          "max_tokens": 2048,
          "extra": {...}
        }

      Response JSON (any of these keys is accepted):
        - {"text": "<output string>"}
        - {"output": "<output string>"}
        - {"content": "<output string>"}

    If you DON'T have a gateway yet, do this (step-by-step):

    1) Create a tiny FastAPI gateway in your infra (or local) that:
       - receives the request above
       - forwards to whatever Cosmos access you have (NVIDIA internal/NIM/etc)
       - returns {"text": "<model output>"}.

    2) Export:
         export COSMOS_BASE_URL="http://localhost:9000"
         export COSMOS_ROUTE="/scene_to_spec"
       Optionally:
         export COSMOS_API_KEY="..."

    3) Then the provider works with your existing DesignAgent without code changes.
    """

    base_url: str
    api_key: Optional[str] = None
    timeout_s: int = 120
    route: str = "/scene_to_spec"


class CosmosProvider:
    """
    Cosmos-backed LLM provider (via your gateway).

    This class is intentionally strict: if COSMOS_BASE_URL is not set and no base_url passed,
    it raises a clear error describing how to wire a gateway.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout_s: Optional[int] = None,
        route: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        base_url = (base_url or os.environ.get("COSMOS_BASE_URL", "")).strip()
        if not base_url:
            raise RuntimeError(
                "CosmosProvider requires COSMOS_BASE_URL (or base_url=...).\n"
                "You need a Cosmos gateway endpoint that accepts JSON and returns a text output.\n"
                "Steps:\n"
                "  1) Implement a small gateway (FastAPI) exposing POST /scene_to_spec.\n"
                "  2) Export COSMOS_BASE_URL='http://<host>:<port>'\n"
                "  3) Optionally export COSMOS_API_KEY if your gateway requires auth.\n"
            )

        api_key = api_key if api_key is not None else os.environ.get("COSMOS_API_KEY", None)
        timeout_s_env = os.environ.get("COSMOS_TIMEOUT_S", "").strip()
        if timeout_s is None:
            timeout_s = int(timeout_s_env) if timeout_s_env else 120

        route = (route or os.environ.get("COSMOS_ROUTE", "/scene_to_spec")).strip()
        if not route.startswith("/"):
            route = "/" + route

        self.cfg = CosmosProviderConfig(
            base_url=base_url.rstrip("/"),
            api_key=(api_key.strip() if isinstance(api_key, str) and api_key.strip() else None),
            timeout_s=int(timeout_s),
            route=route,
        )
        self._session = session or requests.Session()

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
        payload: Dict[str, Any] = {
            "prompt": str(prompt),
            "system": (str(system) if system is not None else None),
            "schema": (str(schema) if schema is not None else None),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "extra": (dict(extra) if extra is not None else {}),
        }

        url = self.cfg.base_url + self.cfg.route
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        r = self._session.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_s)
        try:
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                "CosmosProvider gateway call failed.\n"
                f"URL: {url}\n"
                f"HTTP: {r.status_code}\n"
                f"Body: {r.text[:2000]}\n"
            ) from e

        try:
            data = r.json()
        except Exception as e:
            raise RuntimeError(
                "CosmosProvider expected JSON response from gateway, but got non-JSON.\n"
                f"URL: {url}\n"
                f"Body: {r.text[:2000]}\n"
            ) from e

        for key in ("text", "output", "content"):
            v = data.get(key, None)
            if isinstance(v, str) and v.strip():
                return v

        raise RuntimeError(
            "CosmosProvider gateway returned JSON but none of {text, output, content} was a non-empty string.\n"
            f"Keys: {list(data.keys())}"
        )

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
        return self.generate(
            prompt=prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
