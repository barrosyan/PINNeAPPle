"""Local LLM provider (Ollama) -- so ``pinneapple_llm`` doesn't require a
paid API/external network call at all: run a model entirely on your own
machine, no data leaves it.

Ollama exposes a plain HTTP API (default ``http://localhost:11434``), so
this needs no SDK beyond ``requests`` (already a dependency of
``pinneapple_tools.hpo_experiments``'s GitHub search, so effectively
already present for anyone using this package's research features). This
follows the same "shell out to the real local tool" bridge pattern as
``pinneapple_simulation.external_solvers.cfd_formats.abaqus_reader``'s
``.odb`` bridge and ``pinneapple_blender.render``: :func:`start_server`
launches the real ``ollama`` binary as a subprocess rather than
reimplementing any part of model serving.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional


def start_server(*, host: str = "127.0.0.1", port: int = 11434, wait_s: float = 10.0) -> subprocess.Popen:
    """Launch ``ollama serve`` as a background subprocess (no-op / raises
    if ``ollama`` is not installed -- there is no bundled fallback, same
    reasoning as every other "requires the real local tool" bridge in this
    package).

    Returns the running ``subprocess.Popen`` handle (caller owns its
    lifecycle -- call ``.terminate()`` when done). Blocks up to
    ``wait_s`` seconds polling the server's own ``/api/tags`` endpoint
    before returning, so the caller can start issuing requests
    immediately after this returns without a manual retry loop.
    """
    if shutil.which("ollama") is None:
        raise FileNotFoundError(
            "'ollama' not found on PATH. Install it from https://ollama.com, or if you have it "
            "installed but not on PATH, run `ollama serve` yourself and skip this helper."
        )
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env={"OLLAMA_HOST": f"{host}:{port}"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    import requests

    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/api/tags", timeout=1)
            if r.status_code == 200:
                return proc
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"ollama serve did not respond on {host}:{port} within {wait_s}s")


def ensure_model_pulled(model: str, *, host: str = "127.0.0.1", port: int = 11434, timeout: int = 3600) -> None:
    """``ollama pull <model>`` if it isn't already present locally. Blocks
    until the pull completes (can be slow for a large model the first
    time -- ``timeout`` bounds it)."""
    import requests

    r = requests.get(f"http://{host}:{port}/api/tags", timeout=10)
    r.raise_for_status()
    have = {m["name"] for m in r.json().get("models", [])}
    if model in have or f"{model}:latest" in have:
        return
    proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ollama pull {model} failed:\n{proc.stdout}\n{proc.stderr}")


def call_ollama(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,  # accepted for call-signature symmetry with the other _call_* helpers; unused
    system: str = "",
    *,
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: int = 300,
) -> str:
    """Chat-completion call against a local Ollama server. Raises a clear
    error (not a generic connection-refused traceback) if no server is
    reachable -- call :func:`start_server` first, or run ``ollama serve``
    yourself."""
    import requests

    url = f"http://{host}:{port}/api/chat"
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        r = requests.post(
            url, json={"model": model or "llama3.1", "messages": messages, "stream": False}, timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"could not reach a local Ollama server at {host}:{port} -- call "
            "pinneapple_llm.local_llm.start_server() first, or run `ollama serve` yourself."
        ) from e
    r.raise_for_status()
    return r.json()["message"]["content"]
