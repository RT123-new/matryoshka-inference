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

import math
import re
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
# Capability probe — strict behaviour, every invariant load-bearing.
# --------------------------------------------------------------------------- #

# Endpoint classification. Only the two ``verified_text_*`` states are safe to
# speculate against, and even then only for *surface* (not token-id) identity;
# everything else must fall back to plain generation. Text mode is therefore
# conditional and experimental: use a token-ID backend (``spec.backend``) for
# unconditional equivalence. See ``docs/spec_phase2_results.md``.
CAP_CLASSIC = "verified_text_classic_alignment"   # index i predicts token i (shift 0)
CAP_SHIFTED = "verified_text_shifted_alignment"   # index i predicts token i+1 (shift 1)
CAP_ECHO_IGNORED = "echo_ignored"                 # prompt not echoed back at all
CAP_ECHO_INCOMPLETE = "echo_incomplete"           # scored input not echoed in full
CAP_GENERATED_ONLY = "generated_logprobs_only"    # logprobs cover generated tokens only
CAP_BAD_SHAPE = "unsupported_response_shape"      # missing tokens/top_logprobs/offsets
CAP_MALFORMED_ARRAYS = "malformed_logprob_arrays"  # length mismatch / non-finite logprobs
CAP_PARTIAL_COVERAGE = "partial_continuation_coverage"  # continuation not fully tiled
CAP_INVALID_OFFSETS = "invalid_offsets"           # non-monotonic / overlap / gap / no map
CAP_UNSUPPORTED_OFFSET_UNITS = "unsupported_offset_units"  # byte offsets, not code points
CAP_AMBIGUOUS_ALIGN = "ambiguous_alignment"       # both shifts verify — cannot disambiguate
CAP_BAD_ALIGN = "unsupported_alignment"           # known greedy continuation verifies at no shift
CAP_BONUS_UNAVAILABLE = "bonus_unavailable"       # no present, unambiguous bonus prediction
CAP_UNSUPPORTED_TOKEN_IDENTITY = "unsupported_token_identity"  # byte-fallback surfaces
CAP_NONDETERMINISTIC_POLICY = "nondeterministic_policy"  # response advertises non-greedy decode
CAP_ERROR = "probe_error"

_USABLE = {CAP_CLASSIC, CAP_SHIFTED}

# llama.cpp renders bytes it cannot show as text as ``<0xE2>`` etc. Such a
# "surface" is a token-identity artifact, not literal text: appending it as a
# string would corrupt output, and text scoring cannot tell it from real text.
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


@dataclass
class EndpointCapability:
    status: str
    shift: int | None = None
    echoed: bool = False
    has_prompt_logprobs: bool = False
    offsets_ok: bool = False
    offset_unit: str | None = None
    continuation_verified: bool = False
    bonus_ok: bool = False
    detail: str = ""

    @property
    def usable(self) -> bool:
        # Defence in depth: even a mis-constructed capability is unusable unless
        # every load-bearing invariant holds. offsets_ok / bonus_ok being false
        # can never yield a usable endpoint.
        return (
            self.status in _USABLE
            and self.offsets_ok
            and self.bonus_ok
            and self.continuation_verified
        )


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _logprobs_finite(token_lps: list, tops: list) -> bool:
    """Every candidate/token logprob is a finite number (index 0 may be null)."""
    for i, lp in enumerate(token_lps):
        if lp is None:
            if i == 0:
                continue
            return False
        if not _finite(lp):
            return False
    for i, top in enumerate(tops):
        if top is None:
            if i == 0:
                continue
            return False
        if not isinstance(top, dict) or not top:
            return False
        if not all(_finite(v) for v in top.values()):
            return False
    return True


def _tiles(text_units, surfaces, offsets) -> bool:
    """Do ``surfaces`` tile a prefix of ``text_units`` contiguously from 0?

    One predicate for monotonicity, non-overlap, no gaps and exact surface
    mapping: each surface must sit exactly where the previous one ended, and its
    bytes/chars must match the text there. ``text_units`` and ``surfaces`` are
    both str (code-point units) or both bytes (byte units).
    """
    if not offsets or offsets[0] != 0:
        return False
    cursor = 0
    for surf, off in zip(surfaces, offsets, strict=True):
        if off != cursor:
            return False
        if text_units[off:off + len(surf)] != surf:
            return False
        cursor = off + len(surf)
    return True


def _classify_offsets(text: str, tokens: list, offsets: list) -> tuple[bool, str]:
    """Detect the offset unit and whether the tiling is exact.

    Returns ``(ok, unit)``: ``ok`` is true only for exact code-point tiling.
    Byte offsets are *detected* (so the caller can reject them explicitly rather
    than silently mis-index multibyte text) but never accepted.
    """
    surfaces = [str(s) for s in tokens]
    if _tiles(text, surfaces, offsets):
        return True, "codepoint"
    tb = text.encode("utf-8")
    if _tiles(tb, [s.encode("utf-8") for s in surfaces], offsets):
        return False, "byte"
    return False, "unknown"


def _policy_marker(obj: dict, choice: dict) -> str | None:
    """Reject a response that advertises a non-greedy / unknown decode policy.

    Faithful greedy engines do not report a sampling policy; one that does, with
    a value we cannot account for, must not be trusted to be raw-argmax.
    """
    for src in (obj, choice):
        pol = src.get("generation_policy") or src.get("sampling") or src.get("x_generation_policy")
        if isinstance(pol, dict):
            temp = pol.get("temperature")
            if temp is not None and float(temp) != 0.0:
                return f"response advertises temperature={temp}"
            for field_name in ("grammar", "logit_bias"):
                if pol.get(field_name):
                    return f"response advertises {field_name}"
            unknown = set(pol) - {"temperature", "top_k", "top_p", "min_p"}
            if unknown:
                return f"response advertises unknown generation-policy fields {sorted(unknown)}"
    return None


def classify_scored_choice(choice: dict, prompt: str, sent: str,
                           obj: dict | None = None) -> EndpointCapability:
    """Classify a single scored ``choice`` against every text-mode invariant.

    Pure (no network) so it can be unit-tested directly and reused by
    :func:`probe_endpoint`. ``sent`` is the complete input that was scored
    (prompt + a known raw-argmax continuation); ``prompt`` is its leading
    context. Only :data:`CAP_CLASSIC` / :data:`CAP_SHIFTED` are usable, and only
    when *all* of: complete echo, compatible array lengths, finite candidate
    logprobs, monotonic non-overlapping code-point offsets that map exactly,
    complete contiguous continuation coverage, a single 100%-verifying shift, and
    a present unambiguous bonus — hold at once.
    """
    obj = obj or {}
    prompt_len, sent_len = len(prompt), len(sent)
    text = choice.get("text") or ""
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens")
    offsets = lp.get("text_offset")
    token_lps = lp.get("token_logprobs")
    tops = lp.get("top_logprobs")

    marker = _policy_marker(obj, choice)
    if marker:
        return EndpointCapability(CAP_NONDETERMINISTIC_POLICY, detail=marker)

    # (1) shape present.
    if not tokens or tops is None or offsets is None:
        return EndpointCapability(
            CAP_BAD_SHAPE, echoed=text.startswith(prompt),
            detail="response.logprobs missing tokens/top_logprobs/text_offset "
                   "(e.g. native llama-server returns logprobs.content)")

    # (2) compatible array lengths — tokens, token_logprobs, top_logprobs, offsets.
    n = len(tokens)
    if not (len(offsets) == n and len(tops) == n and token_lps is not None and len(token_lps) == n):
        return EndpointCapability(
            CAP_MALFORMED_ARRAYS, echoed=text.startswith(prompt),
            detail=f"logprob arrays have mismatched lengths (tokens={n}, "
                   f"offsets={len(offsets)}, top={len(tops)}, "
                   f"token_logprobs={None if token_lps is None else len(token_lps)})")

    # (3) echo gates — complete echo of the whole scored input, not just the prompt.
    if not text.startswith(prompt):
        return EndpointCapability(CAP_ECHO_IGNORED, echoed=False,
                                  detail="echo not applied; response text does not include the prompt")
    if not any(o < prompt_len for o in offsets):
        return EndpointCapability(CAP_GENERATED_ONLY, echoed=True, has_prompt_logprobs=False,
                                  detail="prompt echoed but logprobs cover generated positions only")
    if not text.startswith(sent):
        return EndpointCapability(CAP_ECHO_INCOMPLETE, echoed=True, has_prompt_logprobs=True,
                                  detail="scored input not echoed in full (continuation omitted)")

    # (4) finite, valid candidate log probabilities.
    if not _logprobs_finite(token_lps, tops):
        return EndpointCapability(CAP_MALFORMED_ARRAYS, echoed=True, has_prompt_logprobs=True,
                                  detail="non-finite or malformed token/candidate log probabilities")

    # (5) token identity: byte-fallback surfaces cannot be verified as text.
    def _is_byte_fallback(s: Any) -> bool:
        return isinstance(s, str) and bool(_BYTE_FALLBACK_RE.match(s))
    for i, s in enumerate(tokens):
        within = prompt_len <= int(offsets[i]) < sent_len
        if within and _is_byte_fallback(s):
            return EndpointCapability(CAP_UNSUPPORTED_TOKEN_IDENTITY, echoed=True,
                                      has_prompt_logprobs=True,
                                      detail=f"byte-fallback surface {s!r} in continuation; "
                                             "text mode cannot prove token identity")
    for top in tops:
        if isinstance(top, dict) and any(_is_byte_fallback(k) for k in top):
            return EndpointCapability(CAP_UNSUPPORTED_TOKEN_IDENTITY, echoed=True,
                                      has_prompt_logprobs=True,
                                      detail="byte-fallback candidate surface; text mode "
                                             "cannot prove token identity")

    # (6) offsets: known unit, exact contiguous mapping.
    ok, unit = _classify_offsets(text, tokens, offsets)
    if unit == "byte":
        return EndpointCapability(CAP_UNSUPPORTED_OFFSET_UNITS, echoed=True, has_prompt_logprobs=True,
                                  offset_unit="byte",
                                  detail="offsets are UTF-8 byte positions, not code points; "
                                         "loop math assumes code points")
    if not ok:
        return EndpointCapability(CAP_INVALID_OFFSETS, echoed=True, has_prompt_logprobs=True,
                                  detail="offsets are non-monotonic, overlapping, gapped, or do "
                                         "not map to the returned surfaces")

    # (7) complete contiguous coverage of the continuation [prompt_len, sent_len].
    cutpoints = {0} | {int(offsets[i]) + len(str(tokens[i])) for i in range(n)}
    cont_idx = [i for i in range(n)
                if prompt_len <= int(offsets[i]) and int(offsets[i]) + len(str(tokens[i])) <= sent_len]
    if prompt_len not in cutpoints or sent_len not in cutpoints or not cont_idx:
        return EndpointCapability(CAP_PARTIAL_COVERAGE, echoed=True, has_prompt_logprobs=True,
                                  offsets_ok=True, offset_unit="codepoint",
                                  detail="continuation is not tiled by whole tokens "
                                         "(a token straddles a seam or coverage is incomplete)")

    # (8) exactly one alignment shift verifies the whole known continuation.
    verifying = []
    for shift in (0, 1):
        toks_s, _ = _parse_logprobs(choice, shift=shift)
        if cont_idx and all(toks_s[i].is_greedy for i in cont_idx):
            verifying.append(shift)
    if not verifying:
        return EndpointCapability(CAP_BAD_ALIGN, echoed=True, has_prompt_logprobs=True,
                                  offsets_ok=True, offset_unit="codepoint",
                                  detail="known greedy continuation verifies at no shift (0 or 1)")
    if len(verifying) > 1:
        return EndpointCapability(CAP_AMBIGUOUS_ALIGN, echoed=True, has_prompt_logprobs=True,
                                  offsets_ok=True, offset_unit="codepoint",
                                  detail="continuation verifies at BOTH shift 0 and 1; "
                                         "alignment is ambiguous and cannot be trusted")
    shift = verifying[0]

    # (9) a present, unambiguous bonus prediction.
    toks_f, preds_f = _parse_logprobs(choice, shift=shift)
    sr = ScoreResult(tokens=toks_f, shift=shift, predictions=preds_f)
    bonus = sr.greedy_after(sent_len)
    if bonus.surface is None or bonus.ambiguous:
        return EndpointCapability(CAP_BONUS_UNAVAILABLE, echoed=True, has_prompt_logprobs=True,
                                  offsets_ok=True, offset_unit="codepoint", continuation_verified=True,
                                  detail="no present, unambiguous bonus prediction after the input")

    status = CAP_CLASSIC if shift == 0 else CAP_SHIFTED
    return EndpointCapability(
        status=status, shift=shift, echoed=True, has_prompt_logprobs=True,
        offsets_ok=True, offset_unit="codepoint", continuation_verified=True, bonus_ok=True,
        detail=f"known greedy continuation verifies 100% at the single shift {shift}; "
               "surface identity only (not token-id)")


def probe_endpoint(upstream: str, api_key: str, model: str,
                   probe_prompt: str = "The quick brown café jumps über the lazy dog. 1 2 3 4 5 6 7 8",
                   timeout: int = 600) -> EndpointCapability:
    """Classify an endpoint by *behaviour*, so we never speculate blindly.

    Generates a known raw-argmax continuation, scores ``prompt + continuation``,
    and hands the result to :func:`classify_scored_choice`, which enforces every
    text-mode invariant. Only an endpoint that echoes the complete scored input,
    returns well-formed finite logprobs with monotonic code-point offsets that
    tile the continuation, verifies that known continuation 100% at exactly one
    shift, and exposes an unambiguous bonus is reported usable — and even then
    only for *surface* identity. Native ``llama-server`` fails the echo gate and
    is reported unusable rather than silently mis-verified.

    The probe prompt deliberately contains multibyte characters (``café``,
    ``über``): on pure-ASCII text byte and code-point offsets coincide, so an
    endpoint reporting UTF-8 *byte* offsets would pass unnoticed and then
    mis-index real multibyte generation. The multibyte probe forces the offset
    unit to reveal itself.
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
    return classify_scored_choice(choice, probe_prompt, sent, obj=obj)


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
