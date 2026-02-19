from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


JSONLike = Union[str, Dict[str, Any]]


@dataclass(frozen=True)
class GeminiProviderConfig:
    api_base: str = "https://generativelanguage.googleapis.com"
    api_version: str = "v1beta"
    model: str = "models/gemini-2.0-flash"

    api_key: Optional[str] = None
    timeout_s: int = 120

    # Optional gateway mode (legacy)
    gateway_base_url: Optional[str] = None
    gateway_route: str = "/generate"


class GeminiProvider:
    """
    Default: calls Gemini Developer API directly (models.generateContent).

    Supports JSON Mode natively:
      - generationConfig.responseMimeType = "application/json"
      - generationConfig.responseJsonSchema = {...}  (JSON Schema)
    Docs show these fields and examples. :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout_s: Optional[int] = None,
        gateway_base_url: Optional[str] = None,
        gateway_route: Optional[str] = None,
        session: Optional["requests.Session"] = None,
    ):
        if requests is None:
            raise RuntimeError("GeminiProvider requires 'requests'. Install: pip install -e \".[research]\"")

        env_gateway = (os.environ.get("GEMINI_BASE_URL") or "").strip() or None

        self.cfg = GeminiProviderConfig(
            api_base=(api_base or os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com")).rstrip("/"),
            api_version=(api_version or os.environ.get("GEMINI_API_VERSION", "v1beta")).strip("/"),
            model=(model or os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")).strip(),
            api_key=(api_key or os.environ.get("GEMINI_API_KEY") or "").strip() or None,
            timeout_s=int(timeout_s or os.environ.get("GEMINI_TIMEOUT_S", "120")),
            gateway_base_url=(gateway_base_url or env_gateway),
            gateway_route=(gateway_route or os.environ.get("GEMINI_ROUTE", "/generate")).strip() if (gateway_route or os.environ.get("GEMINI_ROUTE") or "/generate") else "/generate",
        )

        if self.cfg.gateway_route and not self.cfg.gateway_route.startswith("/"):
            object.__setattr__(self.cfg, "gateway_route", "/" + self.cfg.gateway_route)

        self._session = session or requests.Session()

    def generate(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[JSONLike] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.cfg.gateway_base_url:
            return self._generate_via_gateway(
                prompt=prompt,
                system=system,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=extra,
            )
        return self._generate_direct(
            prompt=prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    def _generate_direct(
        self,
        *,
        prompt: str,
        system: Optional[str],
        schema: Optional[JSONLike],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]],
    ) -> str:
        if not self.cfg.api_key:
            raise RuntimeError(
                "GeminiProvider direct mode requires GEMINI_API_KEY.\n"
                "Set: export GEMINI_API_KEY='...'\n"
                "Optionally: export GEMINI_MODEL='models/gemini-2.0-flash'"
            )

        model = (extra or {}).get("model") or self.cfg.model
        if not isinstance(model, str) or not model.startswith("models/"):
            raise RuntimeError("Invalid model. Expected 'models/<name>' (example: models/gemini-2.0-flash).")

        # Request: contents[].parts[].text :contentReference[oaicite:2]{index=2}
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }

        # Prefer official systemInstruction field (text only) :contentReference[oaicite:3]{index=3}
        if system:
            body["systemInstruction"] = {"parts": [{"text": str(system)}]}

        # ---- JSON MODE (native) ----
        # If schema is provided, enforce JSON output using responseMimeType + responseJsonSchema.
        # Docs list responseMimeType/responseSchema/responseJsonSchema in generationConfig. :contentReference[oaicite:4]{index=4}
        if schema is not None:
            schema_obj = self._coerce_schema(schema)

            # We send BOTH camelCase and snake_case to be robust across examples/docs.
            # (Docs/examples show both styles in different snippets.) :contentReference[oaicite:5]{index=5}
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseJsonSchema"] = schema_obj

        # passthrough advanced knobs
        if isinstance(extra, dict):
            for k in ("tools", "toolConfig", "safetySettings", "cachedContent"):
                if k in extra:
                    body[k] = extra[k]

        url = f"{self.cfg.api_base}/{self.cfg.api_version}/{model}:generateContent"
        headers = {"Content-Type": "application/json"}

        # Auth: doc examples commonly use ?key=... :contentReference[oaicite:6]{index=6}
        r = self._session.post(
            url,
            params={"key": self.cfg.api_key},
            headers=headers,
            data=json.dumps(body),
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:2000]}")

        data = r.json()
        text = self._extract_text(data)
        if not text:
            raise RuntimeError(f"Gemini API returned no text. keys={list(data.keys())}")
        return text

    @staticmethod
    def _coerce_schema(schema: JSONLike) -> Dict[str, Any]:
        if isinstance(schema, dict):
            return schema
        if isinstance(schema, str):
            s = schema.strip()
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
                # allow list schema root too
                return {"type": "array", "items": obj}
            except Exception as e:
                raise RuntimeError(f"schema string must be valid JSON. error={e}") from e
        raise RuntimeError("schema must be a dict or a JSON string")

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        cands = data.get("candidates") or []
        if not cands:
            return ""
        content = (cands[0].get("content") or {})
        parts = content.get("parts") or []
        out = []
        for p in parts:
            t = p.get("text")
            if isinstance(t, str):
                out.append(t)
        return "".join(out).strip()

    # ---------- Gateway (legacy) ----------
    def _generate_via_gateway(
        self,
        *,
        prompt: str,
        system: Optional[str],
        schema: Optional[JSONLike],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]],
    ) -> str:
        base = (self.cfg.gateway_base_url or "").rstrip("/")
        route = self.cfg.gateway_route if self.cfg.gateway_route.startswith("/") else ("/" + self.cfg.gateway_route)
        url = base + route

        payload: Dict[str, Any] = {
            "prompt": str(prompt),
            "system": (str(system) if system is not None else None),
            "schema": schema,  # gateway may ignore or use it
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "extra": (dict(extra) if extra is not None else {}),
        }
        r = self._session.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Gateway error {r.status_code}: {r.text[:2000]}")
        data = r.json()
        for key in ("text", "output", "content"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        raise RuntimeError(f"Gateway returned no text. keys={list(data.keys())}")
