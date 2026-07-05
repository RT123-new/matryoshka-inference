"""Model-agnostic proxy backend.

Forwards chat completions to any OpenAI-compatible upstream (Ollama at
:11434/v1, LM Studio at :1234/v1, llama.cpp server, vLLM, or a remote API) and
streams the tokens back, so the live dashboard reports real throughput for
*whatever* model is running — not just the Orthrus MLX checkpoints.

We always stream from the upstream (even when the client wants a single
response) so per-token telemetry is accurate; non-streaming clients get the
aggregated result.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

import requests


def discover_ollama_model(base_url: str = "http://localhost:11434") -> Optional[str]:
    """Return the first available Ollama model, or None."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        if r.ok:
            models = r.json().get("models") or []
            if models:
                return models[0]["name"]
    except requests.RequestException:
        pass
    return None


def stream_chat(
    upstream: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: str = "",
    timeout: int = 600,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(kind, data)`` events from the upstream stream.

    kind is "delta" (data = text piece) or "usage" (data = usage dict) or
    "error" (data = message). Always finishes cleanly.
    """
    url = upstream.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # ask upstreams that support it to include usage in the stream
        "stream_options": {"include_usage": True},
    }
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
            if not resp.ok:
                yield "error", f"upstream {resp.status_code}: {resp.text[:200]}"
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    yield "usage", obj["usage"]
                choices = obj.get("choices") or []
                if choices:
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield "delta", delta
    except requests.RequestException as exc:
        yield "error", str(exc)
