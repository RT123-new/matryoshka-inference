"""Scoring client: one round-trip that verifies a whole draft.

The primitive is the OpenAI legacy completions call with ``echo=true`` and
``logprobs=k``: for every position of the text it was sent, the engine returns
the token it saw *and* a small distribution of candidates. For **raw-argmax
greedy** decoding, the top candidate at a position IS the token the engine
would generate there — so scoring ``context + draft`` in one parallel prefill
pass tells us how many draft tokens greedy decoding would have produced, plus
the correct token at the first divergence.

Two hard lessons from testing this against real engines (see
``docs/spec_phase1_results.md``); both are load-bearing:

1. **API shape is not API semantics.** ``llama-cpp-python`` returns the classic
   ``{tokens, token_logprobs, top_logprobs, text_offset}`` object, but its
   ``top_logprobs[i]`` describes the model's distribution *after* token ``i``
   (the prediction for token ``i+1``), i.e. it is shifted by one relative to
   the convention where index ``i`` predicts token ``i``. The prediction that
   *produced* echoed token ``i`` therefore lives at ``top_logprobs[i - shift]``.
   ``shift`` must be measured per endpoint with :func:`probe_endpoint`, never
   assumed — the wrong shift silently drops/duplicates tokens.

2. **Verification must be conservative.** We only accept a draft token when its
   surface is *exactly* the unambiguous top candidate. A tie at the top, a
   missing candidate list, or an empty/unknown surface all force a fallback —
   text-level scoring cannot prove token-*id* identity, so we never guess.

Not every OpenAI-compatible server implements the primitive at all: the current
native ``llama.cpp`` ``llama-server`` ignores ``echo`` and returns logprobs for
generated tokens only, so :func:`probe_endpoint` classifies it unusable and the
loop falls back to plain generation. Engines with prefix caching (llama.cpp
``cache_prompt``, vLLM APC, SGLang RadixAttention) make repeated scoring cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

# Verification reads the engine's *raw* per-position top candidate. For that to
# equal the token generation emits, generation must decode by raw argmax too —
# plain greedy with every history-dependent logit processor disabled. Repetition
# / presence / frequency penalties and top-k/top-p/min-p all make the emitted
# token depend on more than the raw next-token distribution, which the public
# scoring API does not expose. These fields are ignored by engines (and the sim)
# that don't implement them, so pinning them is safe everywhere and keeps the
# scoring lane and the generation lane on one identical policy (llama.cpp and
# llama-cpp-python otherwise default to repeat_penalty=1.1).
_GREEDY_POLICY = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


@dataclass
class Prediction:
    """The engine's greedy candidate for one position, with a safety flag."""
    surface: str | None            # top candidate's text (None if unavailable)
    logprob: float | None
    ambiguous: bool = False        # top tied / missing / unusable → never accept


def _prediction_from_top(top: Any) -> Prediction:
    """Reduce one ``top_logprobs`` entry to a conservative :class:`Prediction`."""
    if not isinstance(top, dict) or not top:
        return Prediction(surface=None, logprob=None, ambiguous=True)
    # Sort by logprob desc; a tie at the very top is unusable (greedy tie-break
    # is by token id, which text-level scoring cannot see).
    ranked = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    surface, logprob = ranked[0]
    ambiguous = len(ranked) >= 2 and ranked[1][1] == logprob
    if surface == "":
        ambiguous = True   # empty/special surface cannot be appended as text
    return Prediction(surface=str(surface), logprob=float(logprob), ambiguous=ambiguous)


@dataclass
class EchoToken:
    surface: str            # the token's text exactly as the engine spells it
    offset: int             # character offset of the token in the scored text
    logprob: float | None        # engine-reported logprob of this token
    top_surface: str | None      # greedy candidate that PRODUCED this position
    top_logprob: float | None    # its logprob
    top_ambiguous: bool = False  # the producing prediction was tied/unusable

    @property
    def is_greedy(self) -> bool:
        """Would greedy decoding *unambiguously* have produced this exact token?

        Conservative by design: exact surface identity against a unique top
        candidate. Ties, missing candidates and empty surfaces are *not*
        greedy — they force the loop to correct or fall back rather than guess.
        We never accept a differently-spelled token because two floating-point
        logprobs happen to print equal.
        """
        return (
            self.top_surface is not None
            and not self.top_ambiguous
            and self.surface == self.top_surface
        )


@dataclass
class ScoreResult:
    tokens: list[EchoToken] = field(default_factory=list)
    text: str = ""                  # echoed prompt + generated tail
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    shift: int = 0
    # Per-response-index greedy predictions (raw, before the shift is applied).
    # predictions[j] is the engine's distribution reported at array index j.
    predictions: list[Prediction] = field(default_factory=list)

    def draft_tokens(self, context_len: int, sent_len: int) -> list[EchoToken] | None:
        """Tokens that cover the draft region ``[context_len, sent_len)``.

        Returns ``None`` when a token straddles the context/draft seam — the
        joint tokenization merged characters across the boundary, so a
        position-wise comparison would be apples-to-oranges and the caller must
        fall back to plain generation for this step.
        """
        picked: list[EchoToken] = []
        for t in self.tokens:
            end = t.offset + len(t.surface)
            if t.offset < context_len < end or t.offset < sent_len < end:
                return None
            if context_len <= t.offset < sent_len:
                picked.append(t)
        return picked

    def greedy_after(self, sent_len: int) -> Prediction:
        """The engine's greedy token for the position right after ``sent``.

        This is the *bonus* token real speculative decoding gets for free from
        its final verify position. It lives at response index
        ``n_echoed - shift``, where ``n_echoed`` is the number of echoed tokens
        that fall inside the text we sent — a unification that is correct for
        both the classic convention (the bonus is the appended generated token)
        and the shifted one (the bonus is the last echoed token's prediction).
        """
        n_echoed = sum(1 for t in self.tokens if t.offset < sent_len)
        idx = n_echoed - self.shift
        if 0 <= idx < len(self.predictions):
            return self.predictions[idx]
        return Prediction(surface=None, logprob=None, ambiguous=True)


def _parse_logprobs(choice: dict, shift: int = 0) -> tuple[list[EchoToken], list[Prediction]]:
    """Parse an echo+logprobs choice under a known positional ``shift``.

    ``shift`` is how far ``top_logprobs``/``token_logprobs`` are shifted from
    the classic convention: 0 means index ``i`` predicts token ``i`` (OpenAI /
    the sim), 1 means index ``i`` predicts token ``i+1`` (llama-cpp-python). The
    prediction that produced echoed token ``i`` is therefore ``predictions[i -
    shift]``. Measure ``shift`` with :func:`probe_endpoint`; do not assume it.
    """
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    token_logprobs = lp.get("token_logprobs") or []
    top_logprobs = lp.get("top_logprobs") or []
    text_offset = lp.get("text_offset") or []
    predictions = [_prediction_from_top(t) for t in top_logprobs]

    out: list[EchoToken] = []
    running = 0
    for i, surface in enumerate(tokens):
        offset = text_offset[i] if i < len(text_offset) else running
        running = offset + len(str(surface))
        logprob = token_logprobs[i] if i < len(token_logprobs) else None
        j = i - shift
        pred = predictions[j] if 0 <= j < len(predictions) else Prediction(None, None, True)
        out.append(EchoToken(
            surface=str(surface), offset=int(offset), logprob=logprob,
            top_surface=pred.surface, top_logprob=pred.logprob,
            top_ambiguous=pred.ambiguous,
        ))
    return out, predictions


def score_completion(upstream: str, api_key: str, model: str, prompt: str,
                     shift: int = 0, logprobs: int = 5, timeout: int = 600) -> ScoreResult:
    """Score ``prompt`` and read one bonus token, all in one round-trip.

    ``max_tokens=1`` (not 0) sidesteps engines that treat 0 as "unlimited". Pass
    the ``shift`` measured by :func:`probe_endpoint` for this endpoint. ``logprobs``
    defaults to 5 so the top-candidate tie check has alternatives to look at.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "echo": True,
        "logprobs": logprobs,
        # Harmless where unsupported; enables KV reuse on llama.cpp server.
        "cache_prompt": True,
        **_GREEDY_POLICY,
    }
    obj, err = _post(upstream, api_key, body, timeout)
    if err:
        return ScoreResult(error=err)
    choice = (obj.get("choices") or [{}])[0]
    text = choice.get("text") or ""
    toks, preds = _parse_logprobs(choice, shift=shift)
    return ScoreResult(
        tokens=toks, text=text, finish_reason=choice.get("finish_reason"),
        usage=obj.get("usage") or {}, shift=shift, predictions=preds,
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
        "cache_prompt": True,
        **_GREEDY_POLICY,
    }
    obj, err = _post(upstream, api_key, body, timeout)
    if err:
        return ScoreResult(error=err)
    choice = (obj.get("choices") or [{}])[0]
    return ScoreResult(
        text=choice.get("text") or "", finish_reason=choice.get("finish_reason"),
        usage=obj.get("usage") or {},
    )


# --------------------------------------------------------------------------- #
# Capability probe — behaviour, not field names.
# --------------------------------------------------------------------------- #

# Endpoint classification. Only the two ``verified_*`` states are safe to
# speculate against; everything else must fall back to plain generation.
CAP_CLASSIC = "verified_classic_alignment"     # index i predicts token i (shift 0)
CAP_SHIFTED = "verified_shifted_alignment"     # index i predicts token i+1 (shift 1)
CAP_ECHO_IGNORED = "echo_ignored"              # prompt not echoed back
CAP_GENERATED_ONLY = "generated_logprobs_only"  # logprobs cover generated tokens only
CAP_BAD_SHAPE = "unsupported_response_shape"   # missing tokens/top_logprobs/offsets
CAP_BAD_ALIGN = "unsupported_alignment"        # a known greedy continuation verifies at no shift
CAP_ERROR = "probe_error"

_USABLE = {CAP_CLASSIC, CAP_SHIFTED}


@dataclass
class EndpointCapability:
    status: str
    shift: int | None = None
    echoed: bool = False
    has_prompt_logprobs: bool = False
    offsets_ok: bool = False
    continuation_verified: bool = False
    bonus_ok: bool = False
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in _USABLE


def _continuation_match_rate(choice: dict, prompt_len: int, sent_len: int,
                             shift: int) -> tuple[int, int]:
    """How many tokens strictly inside the continuation region verify as greedy?"""
    toks, _ = _parse_logprobs(choice, shift=shift)
    good = total = 0
    for t in toks:
        if prompt_len <= t.offset and t.offset + len(t.surface) <= sent_len:
            total += 1
            good += t.is_greedy
    return good, total


def probe_endpoint(upstream: str, api_key: str, model: str,
                   probe_prompt: str = "The quick brown fox jumps over the lazy dog. 1 2 3 4 5 6 7 8",
                   timeout: int = 600) -> EndpointCapability:
    """Classify an endpoint by *behaviour*, so we never speculate blindly.

    Generates a known raw-argmax continuation, scores ``prompt + continuation``,
    and requires the endpoint to prove it (a) echoes the prompt, (b) returns
    prompt-position candidates, (c) has a measurable positional shift under
    which that known greedy continuation verifies end-to-end, and (d) exposes
    the bonus position. Native ``llama-server`` fails at (a)/(b) and is reported
    unusable rather than silently mis-verified.
    """
    cont = generate_burst(upstream, api_key, model, probe_prompt, 16, timeout=timeout)
    if cont.error or not cont.text:
        return EndpointCapability(CAP_ERROR, detail=f"probe generation failed: {cont.error or 'empty'}")
    sent = probe_prompt + cont.text
    body = {"model": model, "prompt": sent, "max_tokens": 1, "echo": True,
            "logprobs": 5, "cache_prompt": True, **_GREEDY_POLICY}
    obj, err = _post(upstream, api_key, body, timeout)
    if err:
        return EndpointCapability(CAP_ERROR, detail=err)
    choice = (obj.get("choices") or [{}])[0]
    text = choice.get("text") or ""
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens") or []
    have_fields = bool(tokens) and lp.get("top_logprobs") and lp.get("text_offset") is not None
    echoed = text.startswith(probe_prompt)
    if not have_fields:
        return EndpointCapability(CAP_BAD_SHAPE, echoed=echoed,
                                  detail="response.logprobs missing tokens/top_logprobs/text_offset "
                                         "(e.g. native llama-server returns logprobs.content)")
    # Echo is the gate: if the returned text does not begin with the prompt, the
    # engine did not score prompt positions and cannot verify drafts.
    if not echoed:
        return EndpointCapability(CAP_ECHO_IGNORED, echoed=False, has_prompt_logprobs=False,
                                  detail="echo not applied; response text does not include the prompt")
    offs = lp.get("text_offset") or []
    if not any(o < len(probe_prompt) for o in offs):
        return EndpointCapability(CAP_GENERATED_ONLY, echoed=echoed, has_prompt_logprobs=False,
                                  detail="prompt echoed but logprobs cover generated positions only")
    # offsets sane: each token's surface appears at its reported offset.
    offsets_ok = all(
        0 <= o <= len(text) and text[o:o + len(str(s))] == str(s)
        for s, o in zip(tokens, offs, strict=False)
    )
    # Which shift makes the KNOWN greedy continuation verify?
    best = None
    for shift in (0, 1):
        good, total = _continuation_match_rate(choice, len(probe_prompt), len(sent), shift)
        if total and (best is None or good / total > best[1]):
            best = (shift, good / total, good, total)
    if best is None or best[3] == 0:
        return EndpointCapability(CAP_BAD_ALIGN, echoed=echoed, has_prompt_logprobs=True,
                                  offsets_ok=offsets_ok, detail="no continuation tokens to verify")
    shift, rate, good, total = best
    if rate < 0.95:
        return EndpointCapability(CAP_BAD_ALIGN, echoed=echoed, has_prompt_logprobs=True,
                                  offsets_ok=offsets_ok,
                                  detail=f"known greedy continuation verifies at only "
                                         f"{good}/{total} (best shift {shift})")
    # Bonus position resolvable?
    _toks, _preds = _parse_logprobs(choice, shift=shift)
    sr = ScoreResult(tokens=_toks, shift=shift, predictions=_preds)
    bonus_ok = sr.greedy_after(len(sent)).surface is not None
    status = CAP_CLASSIC if shift == 0 else CAP_SHIFTED
    return EndpointCapability(
        status=status, shift=shift, echoed=echoed, has_prompt_logprobs=True,
        offsets_ok=offsets_ok, continuation_verified=True, bonus_ok=bonus_ok,
        detail=f"greedy continuation verified {good}/{total} at shift {shift}"
        + ("" if offsets_ok else "; WARNING offsets did not map to surfaces"),
    )


def _post(upstream: str, api_key: str, body: dict, timeout: int) -> tuple[dict, str | None]:
    try:
        resp = requests.post(upstream.rstrip("/") + "/completions",
                             json=body, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as exc:
        return {}, str(exc)
    if not resp.ok:
        return {}, f"upstream {resp.status_code}: {resp.text[:300]}"
    try:
        return resp.json(), None
    except ValueError as exc:
        return {}, f"malformed upstream response: {exc}"


def _headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h
