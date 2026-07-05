"""OpenAI-compatible HTTP server for the Orthrus MLX runtime.

Exposes /v1/chat/completions (streaming + non-streaming) and /v1/models so any
OpenAI-compatible client — Hermes Agent desktop, Open WebUI, LM Studio's remote
provider, curl, the openai SDK — can use the accelerated decoder.

Decoding defaults to the stack this project measured as safest-fastest:
request-level mode routing (structured/reasoning -> diffusion, prose -> AR)
plus the DSpark-style speculation scheduler (backoff 96) inside diffusion mode.
Output is verified by the exact AR pass, so it matches plain decoding.

Stdlib only (http.server); single request at a time, which matches how a local
MLX model actually executes on one GPU.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from sclab.runtimes.orthrus_engine import (
    BlockPolicy,
    ar_generate,
    load_orthrus,
    orthrus_generate,
    route_mode,
)
from sclab.runtimes.orthrus_mlx import MODEL_ALIASES, _resolve_repo
from sclab.telemetry import TelemetryStore
from sclab.dashboard import DASHBOARD_HTML
from sclab import proxy as _proxy

_STATE: dict[str, Any] = {"model": None, "tokenizer": None, "repo": None, "served_name": None,
                          "mode": "auto", "block_size": 16, "backoff": 96,
                          "backend": "orthrus", "upstream": None, "api_key": "", "lock": None}
_TELEMETRY = TelemetryStore()


def _load(repo_id: str) -> None:
    model, tokenizer, _ = load_orthrus(repo_id)
    _STATE.update(model=model, tokenizer=tokenizer, repo=repo_id)


def _messages_to_prompt_ids(messages: list[dict], enable_thinking: bool = False) -> tuple[list[int], str]:
    tok = _STATE["tokenizer"]
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False
    )
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
    return list(tok(text, return_tensors=None)["input_ids"]), user_text


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
        else:
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _handle_proxy(self, body: dict):
        """Faithful pass-through: forward the client's full request to the
        upstream and relay the response verbatim (so tools/tool_calls survive),
        observing the stream only for telemetry."""
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        messages = body.get("messages") or []
        user_text = " ".join(
            m.get("content", "") for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        fwd = dict(body)
        fwd["model"] = body.get("model") or _STATE["model"]  # transparent: keep client's model
        _TELEMETRY.start(rid, "proxy", user_text)

        if body.get("stream"):
            self._sse_open()
            try:
                for kind, data in _proxy.forward_stream(_STATE["upstream"], _STATE["api_key"], fwd):
                    if kind == "raw":
                        self.wfile.write((data + "\n").encode())
                        if data == "":
                            self.wfile.flush()
                    elif kind == "tokens":
                        for _ in range(data):
                            _TELEMETRY.tick(None)
                    elif kind == "error":
                        self.wfile.write(f"data: {json.dumps({'error': {'message': data}})}\n\n".encode())
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                _TELEMETRY.finish(None, 0)
            return

        status, obj = _proxy.forward_once(_STATE["upstream"], _STATE["api_key"], fwd)
        n = int(((obj.get("usage") or {}).get("completion_tokens")) or 0)
        if n:
            _TELEMETRY.set_tokens(n)
        _TELEMETRY.finish(None, 0)
        self._json(status, obj)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
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
        _TELEMETRY.start(rid, mode, user_text)

        if not stream:
            out: list[int] = []
            telemetry = None
            for t, telemetry in gen:
                out.append(t)
                _TELEMETRY.tick(telemetry)
            _TELEMETRY.finish(telemetry, len(prompt_ids))
            text = tok.decode(out)
            if text.endswith(tok.eos_token or ""):
                text = text[: -len(tok.eos_token)]
            self._json(200, {
                "id": rid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text.strip()}}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out),
                          "total_tokens": len(prompt_ids) + len(out)},
                "sclab": (telemetry.summary() | {"decode_mode": mode}) if telemetry else {"decode_mode": mode},
            })
            return

        # --- SSE streaming ------------------------------------------------- #
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta: dict, finish: Optional[str] = None) -> bytes:
            payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        telemetry = None
        try:
            self.wfile.write(chunk({"role": "assistant", "content": ""}))
            # Decode incrementally: emit text as soon as it detokenizes cleanly.
            pending: list[int] = []
            eos_id = tok.eos_token_id
            for t, telemetry in gen:
                _TELEMETRY.tick(telemetry)
                if t == eos_id:
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
            self.wfile.write(chunk({}, finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up mid-stream
        finally:
            _TELEMETRY.finish(telemetry, len(prompt_ids))


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
          served_name: Optional[str] = None, backend: str = "auto",
          upstream: Optional[str] = None, api_key: str = "",
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
        _load(repo)
        _STATE.update(backend="orthrus", mode=mode, block_size=block_size, backoff=backoff,
                      served_name=served_name or model)

    dash = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/dashboard"
    print(f"Serving {_STATE['served_name']} ({backend}) on http://{host}:{port}/v1")
    print(f"Dashboard:  {dash}")
    print(f"Hermes base URL:  http://{host}:{port}/v1   (API key: any)")
    if open_dashboard:
        _open_browser(dash)
    ThreadingHTTPServer((host, port), _Handler).serve_forever()
