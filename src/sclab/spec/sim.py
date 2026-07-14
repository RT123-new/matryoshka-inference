"""Simulated OpenAI-completions engine for the speculation protocol.

Two jobs:

1. **Tests / CI** — a deterministic toy LM with exact ``echo`` + ``logprobs``
   + ``text_offset`` semantics, so the lossless-speculation loop can be proven
   byte-identical against plain generation without any weights or GPU.
2. **Cost-model demos** — an optional latency model (per-request overhead,
   per-token prefill, per-token decode, with llama.cpp-style ``cache_prompt``
   prefix reuse) so ``sclab spec-bench`` can show the *shape* of the win on
   machines with no local engine. Simulated numbers are labeled simulated;
   point spec-bench at a real engine for real ones.

The toy LM ("lag LM") is deliberately simple: the greedy token at position
``p`` is the token at position ``p - lag``. Any prompt therefore continues by
replaying its own earlier content — a crisp stand-in for the copy/template
structure that makes real agent workloads speculable, and fully deterministic
so losslessness is checkable to the byte.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_TOKEN_RE = re.compile(r"\s*\S+")


def tokenize(text: str) -> list[tuple[str, int]]:
    """Whitespace-word tokens with offsets; a leading-space style tokenizer."""
    return [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(text)]


class LagLM:
    """Greedy token at position p == token at position p - lag.

    Every returned token carries a leading space, exactly as real subword
    tokenizers (SentencePiece ``▁``, GPT-2 ``Ġ``) do. That keeps generated
    sequences *canonical*: concatenating whole tokens re-tokenizes to the same
    tokens, so re-feeding output as a prompt is idempotent — the property real
    engines rely on and that a faithful sim must model.
    """

    def __init__(self, lag: int = 12, seed_phrase: str = " the") -> None:
        self.lag = lag
        self.seed = _leading_space(seed_phrase)

    def top1(self, tokens: list[str], p: int) -> str:
        if p - self.lag >= 0:
            return _leading_space(tokens[p - self.lag])
        return self.seed


def _leading_space(tok: str) -> str:
    return tok if tok.startswith(" ") else " " + tok


class SimEngine:
    """Deterministic completions engine with an optional latency model."""

    def __init__(self, lm: LagLM | None = None, overhead_ms: float = 0.0,
                 prefill_ms_per_token: float = 0.0, decode_ms_per_token: float = 0.0,
                 max_total_tokens: int = 100_000, logprob_shift: int = 0) -> None:
        self.lm = lm or LagLM()
        self.overhead_ms = overhead_ms
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.max_total_tokens = max_total_tokens
        # 0 = classic OpenAI convention (index i predicts token i); 1 = the
        # llama-cpp-python convention (index i reports the model's logits *after*
        # token i, i.e. predicts token i+1). Lets tests exercise both alignments
        # deterministically without a real engine.
        self.logprob_shift = logprob_shift
        self._cache_text = ""   # cache_prompt-style single-slot prefix cache
        self._lock = threading.Lock()

    # -- latency model ---------------------------------------------------- #
    def _charge(self, prompt: str, generated_tokens: int) -> None:
        with self._lock:
            shared = 0
            limit = min(len(prompt), len(self._cache_text))
            while shared < limit and prompt[shared] == self._cache_text[shared]:
                shared += 1
            cached_tokens = len(tokenize(prompt[:shared]))
            new_tokens = max(0, len(tokenize(prompt)) - cached_tokens)
            self._cache_text = prompt
        cost = (self.overhead_ms
                + new_tokens * self.prefill_ms_per_token
                + generated_tokens * self.decode_ms_per_token)
        if cost > 0:
            time.sleep(cost / 1000.0)

    # -- completions semantics --------------------------------------------- #
    def complete(self, prompt: str, max_tokens: int, echo: bool, logprobs: int | None):
        # Real tokenizers are idempotent on whole-token model output: sending
        # "prompt + output" back re-tokenizes to the same tokens. This toy
        # tokenizer drops dangling trailing whitespace, so canonicalize the
        # prompt through a tokenize->join round-trip to model that property —
        # otherwise re-feeding output as prompt would spuriously diverge.
        prompt = "".join(s for s, _ in tokenize(prompt))
        surfaces = [t for t, _ in tokenize(prompt)]
        n_prompt = len(surfaces)

        generated: list[str] = []
        finish = "length"
        for i in range(max_tokens):
            if n_prompt + i >= self.max_total_tokens:
                finish = "stop"
                break
            generated.append(self.lm.top1(surfaces + generated, n_prompt + i))
        gen_text = "".join(generated)
        self._charge(prompt, len(generated))

        full_text = (prompt + gen_text) if echo else gen_text
        choice: dict = {"index": 0, "text": full_text, "finish_reason": finish}
        if logprobs is not None:
            # The joint tokenization of prompt+generation, exactly as a real
            # engine would see it (seam merges included).
            full_toks = tokenize(prompt + gen_text)
            all_surfaces = [t for t, _ in full_toks]
            shift = self.logprob_shift
            base = 0 if echo else len(prompt)
            if echo:
                # Classic engines echo the sent tokens *and* the generated tail;
                # the shifted (llama-cpp-python) convention echoes only the sent
                # tokens and carries the bonus in the last position's prediction.
                start_idx = 0
                end_idx = len(full_toks) if shift == 0 else n_prompt
            else:
                start_idx, end_idx = n_prompt, len(full_toks)
            tokens_out: list[str] = []
            offsets: list[int] = []
            lps: list[float | None] = []
            tops: list[dict | None] = []
            for idx in range(start_idx, end_idx):
                surface, offset = full_toks[idx]
                tokens_out.append(surface)
                offsets.append(offset - base)
                if idx == 0:
                    # Both conventions null the very first position.
                    lps.append(None)
                    tops.append(None)
                    continue
                # Response index i reports the model's logits after token
                # (i - shift), i.e. the prediction for position (i - shift) + ...
                # equivalently top_logprobs[i] is the distribution for token
                # i + shift. See verify._parse_logprobs.
                top = self.lm.top1(all_surfaces, idx + shift)
                lps.append(-0.05 if surface == top else -6.0)
                tops.append({top: -0.05})
            choice["logprobs"] = {"tokens": tokens_out, "token_logprobs": lps,
                                  "top_logprobs": tops, "text_offset": offsets}
        usage = {"prompt_tokens": n_prompt, "completion_tokens": len(generated),
                 "total_tokens": n_prompt + len(generated)}
        return choice, usage


def make_handler(engine: SimEngine):
    class _SimHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [{"id": "sim-lag-lm", "object": "model"}]})
            else:
                self._json(200, {"status": "ok", "engine": "sclab-sim"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/completions"):
                self._json(404, {"error": {"message": "unknown path"}})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": {"message": "bad json"}})
                return
            prompt = body.get("prompt") or ""
            if not isinstance(prompt, str):
                self._json(400, {"error": {"message": "prompt must be a string"}})
                return
            choice, usage = engine.complete(
                prompt=prompt,
                max_tokens=int(body.get("max_tokens") or 16),
                echo=bool(body.get("echo")),
                logprobs=body.get("logprobs"),
            )
            self._json(200, {"id": "cmpl-sim", "object": "text_completion",
                             "model": body.get("model") or "sim-lag-lm",
                             "choices": [choice], "usage": usage})

    return _SimHandler


def start_sim_server(engine: SimEngine, host: str = "127.0.0.1", port: int = 0
                     ) -> tuple[ThreadingHTTPServer, str]:
    """Start the sim engine on a background thread; returns (server, base_url)."""
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}/v1"
