"""End-to-end tests for the OpenAI-compatible server in proxy mode.

Spin up a fake OpenAI-compatible upstream and the real sclab handler on
loopback ports, then exercise the proxy through actual HTTP: model mapping,
stream relay, usage-chunk hygiene, and the generic /v1 passthrough. No MLX
or model checkpoints required.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import sclab.server as srv


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[dict] = []  # replaced per fixture instance

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, json.dumps(
                {"object": "list", "data": [{"id": "upmodel", "object": "model"}]}).encode())
        else:
            self._send(404, b'{"error": {"message": "unknown upstream GET"}}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append({"path": self.path, "body": body})
        if self.path.endswith("/chat/completions"):
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                chunks = [
                    {"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
                    {"id": "x", "object": "chat.completion.chunk", "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
                ]
                if (body.get("stream_options") or {}).get("include_usage"):
                    chunks.append({"id": "x", "object": "chat.completion.chunk", "choices": [],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._send(200, json.dumps({
                    "id": "x", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }).encode())
        elif self.path.endswith("/embeddings"):
            self._send(200, json.dumps({
                "object": "list", "model": body.get("model"),
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            }).encode())
        else:
            self._send(404, b'{"error": {"message": "unknown upstream POST"}}')


@pytest.fixture()
def proxy_setup():
    upstream_cls = type("UpstreamForTest", (_Upstream,), {"seen": []})
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), upstream_cls)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}/v1"

    old_state = dict(srv._STATE)
    old_telemetry = srv._TELEMETRY
    srv._STATE.update(backend="proxy", upstream=upstream_url, api_key="",
                      model="upmodel", served_name="upmodel", mode="proxy")
    srv._TELEMETRY = srv.TelemetryStore()

    server = ThreadingHTTPServer(("127.0.0.1", 0), srv._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base, upstream_cls
    finally:
        server.shutdown()
        server.server_close()
        upstream.shutdown()
        upstream.server_close()
        srv._STATE.clear()
        srv._STATE.update(old_state)
        srv._TELEMETRY = old_telemetry


def _sse_events(resp: requests.Response) -> list[str]:
    return [line[5:].strip() for line in resp.iter_lines(decode_unicode=True)
            if line and line.startswith("data:")]


def test_health_and_models(proxy_setup):
    base, _ = proxy_setup
    health = requests.get(f"{base}/health", timeout=5).json()
    assert health["status"] == "ok"
    assert health["backend"] == "proxy"
    models = requests.get(f"{base}/v1/models", timeout=5).json()
    assert models["data"][0]["id"] == "upmodel"


def test_playground_local_model_maps_to_served_model(proxy_setup):
    base, upstream_cls = proxy_setup
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": "local", "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    # The upstream must see the real served model, not the placeholder.
    assert upstream_cls.seen[-1]["body"]["model"] == "upmodel"


def test_real_client_model_passes_through_untouched(proxy_setup):
    base, upstream_cls = proxy_setup
    requests.post(f"{base}/v1/chat/completions", json={
        "model": "gemma4:latest", "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10)
    assert upstream_cls.seen[-1]["body"]["model"] == "gemma4:latest"


def test_stream_withholds_injected_usage_chunk(proxy_setup):
    base, upstream_cls = proxy_setup
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}],
    }, stream=True, timeout=10)
    events = _sse_events(r)
    # The proxy injected include_usage for telemetry...
    assert upstream_cls.seen[-1]["body"]["stream_options"] == {"include_usage": True}
    # ...but the client never asked for it, so no usage-only chunk is relayed.
    parsed = [json.loads(e) for e in events if e != "[DONE]"]
    assert all(obj.get("choices") for obj in parsed)
    assert events[-1] == "[DONE]"
    content = "".join(
        obj["choices"][0].get("delta", {}).get("content") or "" for obj in parsed)
    assert content == "Hello"


def test_stream_relays_usage_chunk_when_client_asked(proxy_setup):
    base, _ = proxy_setup
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": "m", "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "hi"}],
    }, stream=True, timeout=10)
    parsed = [json.loads(e) for e in _sse_events(r) if e != "[DONE]"]
    usage_chunks = [obj for obj in parsed if obj.get("usage") and not obj.get("choices")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"]["completion_tokens"] == 2


def test_unknown_v1_endpoint_passes_through(proxy_setup):
    base, upstream_cls = proxy_setup
    r = requests.post(f"{base}/v1/embeddings", json={"model": "m", "input": "hello"}, timeout=10)
    assert r.status_code == 200
    assert r.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert upstream_cls.seen[-1]["path"].endswith("/v1/embeddings")


def test_telemetry_counts_streamed_request(proxy_setup):
    base, _ = proxy_setup
    resp = requests.post(f"{base}/v1/chat/completions", json={
        "model": "m", "stream": True, "messages": [{"role": "user", "content": "count me"}],
    }, stream=True, timeout=10)
    _ = resp.content  # drain the stream so the server-side finish() has run
    stats = requests.get(f"{base}/dashboard/stats", timeout=5).json()
    assert stats["totals"]["requests"] == 1
    # exact count from the upstream usage chunk, not the per-chunk heuristic
    assert stats["totals"]["tokens"] == 2
    assert stats["history"][0]["prompt_preview"] == "count me"


def test_content_parts_user_text_reaches_preview(proxy_setup):
    base, _ = proxy_setup
    requests.post(f"{base}/v1/chat/completions", json={
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "part style"}]}],
    }, timeout=10)
    stats = requests.get(f"{base}/dashboard/stats", timeout=5).json()
    assert stats["history"][0]["prompt_preview"] == "part style"


def test_cors_preflight(proxy_setup):
    base, _ = proxy_setup
    r = requests.options(f"{base}/v1/chat/completions", timeout=5)
    assert r.status_code == 204
    assert r.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in r.headers["Access-Control-Allow-Methods"]
