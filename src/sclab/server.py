"""OpenAI-compatible HTTP server for the Orthrus MLX runtime.

Exposes /v1/chat/completions (streaming + non-streaming) and /v1/models so any
OpenAI-compatible client — Hermes Agent desktop, Open WebUI, LM Studio's remote
provider, curl, the openai SDK — can use the accelerated decoder. In proxy mode
every other /v1 endpoint (embeddings, legacy completions, ...) is relayed to the
upstream byte-for-byte, so the transparent proxy never 404s a request the real
endpoint would have answered.

Decoding defaults to the stack this project measured as safest-fastest:
request-level mode routing (structured/reasoning -> diffusion, prose -> AR)
plus the DSpark-style speculation scheduler (backoff 96) inside diffusion mode.
Output is verified by the exact AR pass, so it matches plain decoding.

Stdlib only (http.server). Requests are handled on threads, but Orthrus
generation is serialised by a lock: one MLX decode at a time matches how the
GPU actually executes and keeps interleaved requests from corrupting decode
state. Proxy requests run concurrently — the upstream handles its own queue.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sclab import proxy as _proxy
from sclab.dashboard import DASHBOARD_HTML
from sclab.runtimes.orthrus_engine import (
    BlockPolicy,
    ar_generate,
    load_orthrus,
    orthrus_generate,
    route_mode,
)
from sclab.runtimes.orthrus_mlx import _resolve_repo
from sclab.telemetry import TelemetryStore

_STATE: dict[str, Any] = {"model": None, "tokenizer": None, "repo": None, "served_name": None,
                          "mode": "auto", "block_size": 16, "backoff": 96,
                          "backend": "orthrus", "upstream": None, "api_key": ""}
_TELEMETRY = TelemetryStore()
# MLX decode is single-GPU and its lazy evaluation is not safe to interleave:
# serialise Orthrus generation across handler threads.
_GEN_LOCK = threading.Lock()


def _load(repo_id: str) -> None:
    model, tokenizer, _ = load_orthrus(repo_id)
    _STATE.update(model=model, tokenizer=tokenizer, repo=repo_id)


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI message ``content`` field to plain text.

    Clients may send a string or a list of content parts
    (``[{"type": "text", "text": ...}, ...]``); non-text parts are skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return ""


def _user_text(messages: list[dict]) -> str:
    return " ".join(
        _content_to_text(m.get("content")) for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


def _messages_to_prompt_ids(messages: list[dict], enable_thinking: bool = False) -> tuple[list[int], str]:
    tok = _STATE["tokenizer"]
    normalized = [
        {**m, "content": _content_to_text(m.get("content"))}
        for m in messages if isinstance(m, dict)
    ]
    text = tok.apply_chat_template(
        normalized, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False
    )
    return list(tok(text, return_tensors=None)["input_ids"]), _user_text(normalized)


def _make_generator(prompt_ids: list[int], user_text: str, max_tokens: int, temperature: float):
    model, tok = _STATE["model"], _STATE["tokenizer"]
    eos = tok.eos_token_id
    mode = _STATE["mode"]
    if mode == "auto":
        mode, _ = route_mode(user_text)
    if mode == "ar":
        return ar_generate(model, prompt_ids, eos, max_tokens, temperature), "ar"
    policy = BlockPolicy(
        mode="scheduled",
        block_size=8,
        min_block=2,
        max_block=_STATE["block_size"],
        structured_block=_STATE["block_size"],
        probe_block=6,
        backoff_steps=_STATE["backoff"],
    )
    gen = orthrus_generate(
        model, prompt_ids, eos, max_tokens, temperature,
        policy=policy, detokenize=lambda ts: tok.decode(ts),
    )
    return gen, "diffusion"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet default logging, keep errors
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str) -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        # CORS preflight, so browser-based OpenAI clients can call the API.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            if _STATE["backend"] == "proxy":
                upstream_models = _proxy.list_models(_STATE["upstream"], _STATE["api_key"])
                if upstream_models is not None:
                    self._json(200, upstream_models)
                    return
            name = _STATE["served_name"] or _STATE["repo"] or _STATE["model"]
            self._json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "created": int(time.time()), "owned_by": "local"}
            ]})
        elif path in ("/dashboard", "/ui"):
            self._html(DASHBOARD_HTML)
        elif path in ("/dashboard/stats", "/stats"):
            snap = _TELEMETRY.snapshot()
            snap["server"] = {"model": _STATE["served_name"] or _STATE["repo"] or _STATE["model"],
                              "mode": _STATE["mode"], "backend": _STATE["backend"],
                              "block_size": _STATE["block_size"], "upstream": _STATE["upstream"]}
            self._json(200, snap)
        elif path in ("", "/health"):
            self._json(200, {"status": "ok", "backend": _STATE["backend"],
                             "model": _STATE["served_name"] or _STATE["repo"] or _STATE["model"],
                             "mode": _STATE["mode"], "dashboard": "/dashboard"})
        elif _STATE["backend"] == "proxy":
            self._relay_upstream("GET", body=None)
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

    def _relay_upstream(self, method: str, body: bytes | None) -> None:
        """Pass any other endpoint through to the upstream verbatim."""
        # Clients speak OpenAI paths (/v1/...); the upstream base already ends
        # in /v1, so strip the prefix before joining.
        path = self.path[3:] if self.path.startswith("/v1/") else self.path
        status, ctype, chunks = _proxy.forward_raw(
            _STATE["upstream"], _STATE["api_key"], method, path,
            body=body, content_type=self.headers.get("Content-Type", "application/json"),
        )
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in chunks:
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_proxy(self, body: dict):
        """Faithful pass-through: forward the client's full request to the
        upstream and relay the response verbatim (so tools/tool_calls survive),
        observing the stream only for telemetry."""
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        messages = body.get("messages") or []
        user_text = _user_text([m for m in messages if isinstance(m, dict)])
        fwd = dict(body)
        # Transparent: keep the client's model. "local" is the dashboard
        # playground's placeholder for "whatever this server serves".
        client_model = body.get("model")
        fwd["model"] = _STATE["model"] if client_model in (None, "", "local") else client_model
        # forward_stream injects include_usage for exact telemetry; remember
        # whether the client asked for it so injected usage chunks can be
        # withheld from a client that never requested them.
        client_wants_usage = "stream_options" in body
        _TELEMETRY.start(rid, "proxy", user_text)

        if body.get("stream"):
            self._sse_open()
            try:
                for kind, data in _proxy.forward_stream(_STATE["upstream"], _STATE["api_key"], fwd):
                    if kind == "raw":
                        if not client_wants_usage and _is_usage_only_chunk(data):
                            continue
                        self.wfile.write((data + "\n").encode())
                        if data == "":
                            self.wfile.flush()
                    elif kind == "tokens":
                        if data:
                            _TELEMETRY.tick(rid, n=data)
                    elif kind == "usage":
                        _TELEMETRY.set_tokens(rid, data)
                    elif kind == "error":
                        self.wfile.write(f"data: {json.dumps({'error': {'message': data}})}\n\n".encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                _TELEMETRY.finish(rid)
            return

        status, obj = _proxy.forward_once(_STATE["upstream"], _STATE["api_key"], fwd)
        usage = obj.get("usage") or {}
        n = int(usage.get("completion_tokens") or 0)
        if n:
            _TELEMETRY.set_tokens(rid, n)
        _TELEMETRY.finish(rid, prompt_tokens=int(usage.get("prompt_tokens") or 0))
        self._json(status, obj)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            if _STATE["backend"] == "proxy":
                length = int(self.headers.get("Content-Length", 0) or 0)
                self._relay_upstream("POST", body=self.rfile.read(length) if length else None)
            else:
                self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            messages = req.get("messages") or []
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 1024)
            temperature = float(req.get("temperature") or 0.0)
            stream = bool(req.get("stream"))
        except Exception as exc:  # malformed request must not kill the server
            self._json(400, {"error": {"message": str(exc)}})
            return

        if _STATE["backend"] == "proxy":
            self._handle_proxy(req)
            return

        with _GEN_LOCK:
            self._generate_orthrus(req, messages, max_tokens, temperature, stream)

    def _generate_orthrus(self, req: dict, messages: list, max_tokens: int,
                          temperature: float, stream: bool) -> None:
        try:
            prompt_ids, user_text = _messages_to_prompt_ids(messages)
            gen, mode = _make_generator(prompt_ids, user_text, max_tokens, temperature)
        except Exception as exc:
            self._json(400, {"error": {"message": str(exc)}})
            return

        tok = _STATE["tokenizer"]
        name = _STATE["served_name"] or _STATE["repo"]
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        eos_id = tok.eos_token_id
        _TELEMETRY.start(rid, mode, user_text)

        if not stream:
            out: list[int] = []
            telemetry = None
            finish_reason = "length"
            try:
                for t, telemetry in gen:
                    _TELEMETRY.tick(rid, telemetry)
                    if t == eos_id:
                        finish_reason = "stop"
                        break
                    out.append(t)
            except Exception as exc:
                _TELEMETRY.finish(rid, telemetry, len(prompt_ids))
                self._json(500, {"error": {"message": f"generation failed: {exc}"}})
                return
            _TELEMETRY.finish(rid, telemetry, len(prompt_ids))
            text = tok.decode(out) if out else ""
            completion_tokens = len(out) + (1 if finish_reason == "stop" else 0)
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": text.strip()}}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": completion_tokens,
                          "total_tokens": len(prompt_ids) + completion_tokens},
                "sclab": (telemetry.summary() | {"decode_mode": mode}) if telemetry else {"decode_mode": mode},
            })
            return

        # --- SSE streaming ------------------------------------------------- #
        self._sse_open()

        def chunk(delta: dict, finish: str | None = None) -> bytes:
            payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        telemetry = None
        n_tokens = 0
        finish_reason = "length"
        try:
            self.wfile.write(chunk({"role": "assistant", "content": ""}))
            # Decode incrementally: emit text as soon as it detokenizes cleanly.
            pending: list[int] = []
            for t, telemetry in gen:
                _TELEMETRY.tick(rid, telemetry)
                n_tokens += 1
                if t == eos_id:
                    finish_reason = "stop"
                    break
                pending.append(t)
                text = tok.decode(pending)
                if not text.endswith("�"):  # incomplete UTF-8 sequence
                    self.wfile.write(chunk({"content": text}))
                    self.wfile.flush()
                    pending = []
            if pending:
                text = tok.decode(pending)
                if not text.endswith("�"):
                    self.wfile.write(chunk({"content": text}))
            self.wfile.write(chunk({}, finish=finish_reason))
            if (req.get("stream_options") or {}).get("include_usage"):
                usage_payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                                 "model": name, "choices": [],
                                 "usage": {"prompt_tokens": len(prompt_ids),
                                           "completion_tokens": n_tokens,
                                           "total_tokens": len(prompt_ids) + n_tokens}}
                self.wfile.write(f"data: {json.dumps(usage_payload)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up mid-stream
        except Exception as exc:
            # Surface generation failures to the client instead of dying silently.
            try:
                self.wfile.write(f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            _TELEMETRY.finish(rid, telemetry, len(prompt_ids))


def _is_usage_only_chunk(line: str) -> bool:
    """True for an SSE data line that carries only usage (no choices) — the
    shape of the chunk include_usage appends."""
    if not line.startswith("data:"):
        return False
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return False
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return False
    return bool(obj.get("usage")) and not obj.get("choices")


def _open_browser(url: str) -> None:
    import threading
    import webbrowser

    def _go():
        import time as _t
        _t.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def serve(model: str, host: str = "127.0.0.1", port: int = 8977,
          mode: str = "auto", block_size: int = 16, backoff: int = 96,
          served_name: str | None = None, backend: str = "auto",
          upstream: str | None = None, api_key: str = "",
          open_dashboard: bool = False) -> None:
    # Resolve backend: explicit upstream (or non-orthrus model) => proxy.
    if backend == "auto":
        backend = "proxy" if upstream else "orthrus"

    if backend == "proxy":
        if not upstream:
            raise SystemExit("proxy backend requires --upstream (e.g. http://localhost:11434/v1)")
        _STATE.update(backend="proxy", upstream=upstream, api_key=api_key,
                      model=model, served_name=served_name or model, mode="proxy")
        print(f"Proxying '{model}' via {upstream}")
    else:
        repo = _resolve_repo(model)
        print(f"Loading {repo} ...")
        try:
            _load(repo)
        except ImportError as exc:
            raise SystemExit(
                f"The Orthrus backend needs MLX + transformers ({exc}).\n"
                "  - On Apple Silicon: pip install -e '.[orthrus]'\n"
                "  - Anywhere else: use the model-agnostic proxy instead, e.g.\n"
                "      sclab serve --backend proxy --upstream http://localhost:11434/v1"
            ) from exc
        _STATE.update(backend="orthrus", mode=mode, block_size=block_size, backoff=backoff,
                      served_name=served_name or model)

    try:
        httpd = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not bind {host}:{port} ({exc}).\n"
            f"Is another sclab serve already running? Try --port {port + 1} "
            f"or stop the other instance."
        ) from exc

    dash = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/dashboard"
    print(f"Serving {_STATE['served_name']} ({backend}) on http://{host}:{port}/v1")
    print(f"Dashboard:  {dash}")
    print(f"Hermes base URL:  http://{host}:{port}/v1   (API key: any)")
    if open_dashboard:
        _open_browser(dash)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
