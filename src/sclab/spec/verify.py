"""Scoring client: one round-trip that verifies a whole draft.

The universal primitive is the OpenAI legacy completions call with
``echo=true`` and ``logprobs=k``: the engine returns, for every position of
the prompt it was sent, the token it actually saw *and* the most likely token
at that position. For greedy decoding, "the most likely token" IS the token
the engine would have generated — so scoring ``context + draft`` in one
parallel prefill pass tells us exactly how many draft tokens greedy decoding
would have produced, plus the correction token at the first divergence.

Engines known to implement this primitive: llama.cpp / llama-cpp-python
(``logits_all``), vLLM (``echo`` + ``prompt_logprobs``), and anything else
that faithfully mirrors the legacy completions API. Engines with automatic
prefix caching (vLLM APC, llama.cpp ``cache_prompt``, SGLang RadixAttention)
make repeated scoring cheap: each round only prefills the new draft tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class EchoToken:
    surface: str            # the token's text exactly as the engine spells it
    offset: int             # character offset of the token in the scored text
    logprob: float | None        # logprob of this token (None for position 0)
    top_surface: str | None      # most likely token at this position
    top_logprob: float | None    # its logprob

    @property
    def is_greedy(self) -> bool:
        """Would greedy decoding have produced this exact token here?"""
        if self.top_surface is None:
            return False
        if self.surface == self.top_surface:
            return True
        # Tie: the engine reports a different spelling with identical mass.
        return self.logprob is not None and self.logprob == self.top_logprob


@dataclass
class ScoreResult:
    tokens: list[EchoToken] = field(default_factory=list)
    text: str = ""                  # echoed prompt + generated tail
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def draft_tokens(self, context_len: int, sent_len: int) -> list[EchoToken] | None:
        """Tokens that cover the draft region [context_len, sent_len).

        Returns ``None`` when a token straddles the context/draft seam —
        the joint tokenization merged context and draft characters into one
        token, so position-wise greedy comparison would be apples-to-oranges
        and the caller must fall back to plain generation for this step.
        """
        picked: list[EchoToken] = []
        for t in self.tokens:
            end = t.offset + len(t.surface)
            if t.offset < context_len < end or t.offset < sent_len < end:
                return None
            if context_len <= t.offset < sent_len:
                picked.append(t)
        return picked

    def generated_tokens(self, sent_len: int) -> list[EchoToken]:
        """Tokens the engine generated beyond the text we sent."""
        return [t for t in self.tokens if t.offset >= sent_len]


def _parse_logprobs(choice: dict, base_text: str) -> list[EchoToken]:
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    token_logprobs = lp.get("token_logprobs") or []
    top_logprobs = lp.get("top_logprobs") or []
    text_offset = lp.get("text_offset") or []
    out: list[EchoToken] = []
    running = 0
    for i, surface in enumerate(tokens):
        offset = text_offset[i] if i < len(text_offset) else running
        running = offset + len(surface)
        logprob = token_logprobs[i] if i < len(token_logprobs) else None
        top = top_logprobs[i] if i < len(top_logprobs) else None
        top_surface = None
        top_lp = None
        if isinstance(top, dict) and top:
            top_surface, top_lp = max(top.items(), key=lambda kv: kv[1])
        out.append(EchoToken(surface=str(surface), offset=int(offset),
                             logprob=logprob, top_surface=top_surface, top_logprob=top_lp))
    return out


def score_completion(upstream: str, api_key: str, model: str, prompt: str,
                     timeout: int = 600) -> ScoreResult:
    """Score ``prompt`` and generate one bonus token, all in one round-trip.

    ``max_tokens=1`` (not 0) both sidesteps engines that treat 0 as
    "unlimited" and gives us the next greedy token for free when the whole
    draft is accepted — the same bonus token real speculative decoding gets
    from its final verify position.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
        # Harmless where unsupported; enables KV reuse on llama.cpp server.
        "cache_prompt": True,
    }
    try:
        resp = requests.post(upstream.rstrip("/") + "/completions",
                             json=body, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as exc:
        return ScoreResult(error=str(exc))
    if not resp.ok:
        return ScoreResult(error=f"upstream {resp.status_code}: {resp.text[:300]}")
    try:
        obj = resp.json()
        choice = (obj.get("choices") or [{}])[0]
    except (ValueError, IndexError) as exc:
        return ScoreResult(error=f"malformed upstream response: {exc}")
    text = choice.get("text") or ""
    return ScoreResult(
        tokens=_parse_logprobs(choice, text),
        text=text,
        finish_reason=choice.get("finish_reason"),
        usage=obj.get("usage") or {},
    )


def generate_burst(upstream: str, api_key: str, model: str, prompt: str,
                   max_tokens: int, timeout: int = 600) -> ScoreResult:
    """Plain greedy generation for when there is nothing worth speculating.

    This is exactly what a non-accelerated client would do, so the burst path
    is lossless by construction and costs nothing extra.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": True,
    }
    try:
        resp = requests.post(upstream.rstrip("/") + "/completions",
                             json=body, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as exc:
        return ScoreResult(error=str(exc))
    if not resp.ok:
        return ScoreResult(error=f"upstream {resp.status_code}: {resp.text[:300]}")
    try:
        obj = resp.json()
        choice = (obj.get("choices") or [{}])[0]
    except (ValueError, IndexError) as exc:
        return ScoreResult(error=f"malformed upstream response: {exc}")
    return ScoreResult(
        text=choice.get("text") or "",
        finish_reason=choice.get("finish_reason"),
        usage=obj.get("usage") or {},
    )


def _headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h
