"""Model-agnostic proxy backend.

Forwards chat completions to any OpenAI-compatible upstream (Ollama at
:11434/v1, LM Studio at :1234/v1, llama.cpp server, vLLM, or a remote API) and
streams the tokens back, so the live dashboard reports real throughput for
*whatever* model is running — not just the Orthrus MLX checkpoints.

The proxy is **faithful**: it forwards the client's full request body (tools,
tool_choice, response_format, stop, seed, …) and passes the upstream's response
through verbatim — so agent features like tool calling survive untouched. It
only observes the stream for telemetry; it never rewrites it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests


def discover_ollama_model(base_url: str = "http://localhost:11434") -> str | None:
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


def list_models(upstream: str, api_key: str, timeout: int = 5) -> dict | None:
    """Return the upstream's /v1/models response verbatim, or None on failure."""
    try:
        r = requests.get(upstream.rstrip("/") + "/models", headers=_headers(api_key), timeout=timeout)
        if r.ok:
            return r.json()
    except (requests.RequestException, ValueError):
        pass
    return None


def _headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def count_delta_tokens(obj: dict) -> int:
    """How many token-ish pieces a streamed chunk carries (content or tool args)."""
    n = 0
    for ch in obj.get("choices") or []:
        delta = ch.get("delta") or {}
        if delta.get("content"):
            n += 1
        for tc in delta.get("tool_calls") or []:
            if (tc.get("function") or {}).get("arguments"):
                n += 1
    return n


def forward_stream(upstream: str, api_key: str, body: dict, timeout: int = 600
                   ) -> Iterator[tuple[str, Any]]:
    """Stream from the upstream, yielding events for the server to relay + count.

    Yields:
      ("raw", line)   - a verbatim SSE data line to write straight to the client
      ("tokens", n)   - token count parsed from that line (for telemetry)
      ("usage", n)    - exact completion_tokens from the upstream's usage chunk
      ("error", msg)  - upstream failure
    The full client body is forwarded; only stream flags are forced on.
    """
    url = upstream.rstrip("/") + "/chat/completions"
    payload = dict(body)
    payload["stream"] = True
    payload.setdefault("stream_options", {"include_usage": True})
    try:
        with requests.post(url, json=payload, headers=_headers(api_key),
                           stream=True, timeout=timeout) as resp:
            if not resp.ok:
                yield "error", f"upstream {resp.status_code}: {resp.text[:300]}"
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                yield "raw", raw
                if raw.startswith("data:"):
                    data = raw[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        yield "tokens", count_delta_tokens(obj)
                        usage = obj.get("usage") or {}
                        if usage.get("completion_tokens"):
                            yield "usage", int(usage["completion_tokens"])
    except requests.RequestException as exc:
        yield "error", str(exc)


def forward_once(upstream: str, api_key: str, body: dict, timeout: int = 600
                 ) -> tuple[int, dict]:
    """Non-streaming forward: return (status_code, json_body) verbatim."""
    url = upstream.rstrip("/") + "/chat/completions"
    payload = dict(body)
    payload["stream"] = False
    try:
        resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=timeout)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": {"message": resp.text[:300]}}
    except requests.RequestException as exc:
        return 502, {"error": {"message": str(exc)}}


def forward_raw(upstream: str, api_key: str, method: str, path: str,
                body: bytes | None = None, content_type: str = "application/json",
                timeout: int = 600) -> tuple[int, str, Iterator[bytes]]:
    """Byte-for-byte passthrough for any other endpoint the upstream serves
    (embeddings, legacy completions, ...), so a client pointed at the proxy
    never hits a 404 the real endpoint would have answered.

    ``path`` is relative to the upstream base (e.g. "/embeddings"). Returns
    ``(status, content_type, chunk_iterator)``; the iterator keeps the upstream
    connection open until drained, and streams SSE bodies chunk-by-chunk.
    """
    url = upstream.rstrip("/") + path
    headers = _headers(api_key)
    headers["Content-Type"] = content_type
    try:
        resp = requests.request(method, url, data=body, headers=headers,
                                stream=True, timeout=timeout)
    except requests.RequestException as exc:
        payload = json.dumps({"error": {"message": str(exc)}}).encode()
        return 502, "application/json", iter([payload])

    def _chunks() -> Iterator[bytes]:
        try:
            yield from resp.iter_content(chunk_size=8192)
        finally:
            resp.close()

    return resp.status_code, resp.headers.get("Content-Type", "application/json"), _chunks()
