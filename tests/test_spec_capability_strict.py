"""Adversarial tests for the strict text-mode capability probe.

Phase 1's probe accepted an endpoint on a 95% continuation match and did not let
``offsets_ok``/``bonus_ok`` being false block a "verified" status — so an
endpoint that mis-indexes multibyte text, omits the bonus, or only partially
covers the continuation could still be trusted. These tests pin the hardened
contract: an endpoint is usable *only* when every load-bearing invariant holds,
and every incompatible endpoint is classified unusable and yields plain
generation only.

Most cases drive :func:`classify_scored_choice` directly (pure, no network) with
hand-built payloads, which is exactly what the probe feeds it; a few spin up a
real HTTP sim to prove the end-to-end fallback.
"""

from __future__ import annotations

import copy

import pytest

from sclab.spec.backend import policy_is_deterministic
from sclab.spec.loop import spec_generate
from sclab.spec.sim import LagLM, SimEngine, start_sim_server
from sclab.spec.verify import (
    CAP_AMBIGUOUS_ALIGN,
    CAP_BAD_ALIGN,
    CAP_BAD_SHAPE,
    CAP_BONUS_UNAVAILABLE,
    CAP_CLASSIC,
    CAP_ECHO_IGNORED,
    CAP_ECHO_INCOMPLETE,
    CAP_INVALID_OFFSETS,
    CAP_MALFORMED_ARRAYS,
    CAP_NONDETERMINISTIC_POLICY,
    CAP_PARTIAL_COVERAGE,
    CAP_UNSUPPORTED_OFFSET_UNITS,
    CAP_UNSUPPORTED_TOKEN_IDENTITY,
    classify_scored_choice,
    generate_burst,
    probe_endpoint,
)

# --------------------------------------------------------------------------- #
# Payload builders. Leading-space tokens tile contiguously from 0.
# --------------------------------------------------------------------------- #

PROMPT = "A B C"          # len 5
SENT = "A B C D E"        # len 9 (prompt + " D E")


def _offsets(tokens):
    offs, cur = [], 0
    for t in tokens:
        offs.append(cur)
        cur += len(t)
    return offs


def _choice(tokens, tops, *, text=None, token_lps=None, offsets=None):
    return {
        "text": "".join(tokens) if text is None else text,
        "logprobs": {
            "tokens": tokens,
            "text_offset": _offsets(tokens) if offsets is None else offsets,
            "token_logprobs": ([None] + [-0.1] * (len(tokens) - 1)) if token_lps is None else token_lps,
            "top_logprobs": tops,
        },
    }


def good_classic():
    """A fully valid classic (shift 0) scored choice: usable, surface identity."""
    tokens = ["A", " B", " C", " D", " E", " F"]   # " F" is the appended bonus token
    tops = [None, {" B": -0.1}, {" C": -0.1}, {" D": -0.1}, {" E": -0.1}, {" F": -0.1}]
    return _choice(tokens, tops, text="A B C D E F")


def test_good_classic_is_usable():
    cap = classify_scored_choice(good_classic(), PROMPT, SENT)
    assert cap.status == CAP_CLASSIC
    assert cap.usable
    assert cap.offsets_ok and cap.bonus_ok and cap.continuation_verified
    assert cap.offset_unit == "codepoint"


# --------------------------------------------------------------------------- #
# One adversarial mutation per invariant. Each must be unusable.
# --------------------------------------------------------------------------- #

def _echo_ignored():
    # response text does not contain the prompt at all (native llama-server).
    ch = good_classic()
    ch["text"] = "totally different text here"
    return ch, PROMPT, SENT, CAP_ECHO_IGNORED


def _echo_incomplete():
    # prompt echoed, continuation omitted from the returned text.
    tokens = ["A", " B", " C"]
    tops = [None, {" B": -0.1}, {" C": -0.1}]
    return _choice(tokens, tops, text="A B C"), PROMPT, SENT, CAP_ECHO_INCOMPLETE


def _partial_coverage():
    # full echo, but only one continuation token is scored (rest omitted).
    tokens = ["A", " B", " C", " D"]
    tops = [None, {" B": -0.1}, {" C": -0.1}, {" D": -0.1}]
    return _choice(tokens, tops, text="A B C D E F"), PROMPT, SENT, CAP_PARTIAL_COVERAGE


def _missing_continuation_position():
    # a continuation token is dropped, leaving a gap in the offsets.
    tokens = ["A", " B", " C", " E", " F"]
    offsets = [0, 1, 3, 7, 9]
    tops = [None, {" B": -0.1}, {" C": -0.1}, {" E": -0.1}, {" F": -0.1}]
    return _choice(tokens, tops, text="A B C D E F", offsets=offsets), PROMPT, SENT, CAP_INVALID_OFFSETS


def _nonmonotonic_offsets():
    ch = good_classic()
    ch["logprobs"]["text_offset"] = [0, 3, 1, 5, 7, 9]
    return ch, PROMPT, SENT, CAP_INVALID_OFFSETS


def _overlapping_offsets():
    ch = good_classic()
    ch["logprobs"]["text_offset"] = [0, 1, 2, 5, 7, 9]
    return ch, PROMPT, SENT, CAP_INVALID_OFFSETS


def _gapped_offsets():
    ch = good_classic()
    ch["logprobs"]["text_offset"] = [0, 1, 4, 6, 8, 10]
    return ch, PROMPT, SENT, CAP_INVALID_OFFSETS


def _missing_bonus():
    # continuation tiles [prompt, sent] and verifies, but there is no bonus
    # prediction past the input (classic without the appended generated token).
    tokens = ["A", " B", " C", " D", " E"]
    tops = [None, {" B": -0.1}, {" C": -0.1}, {" D": -0.1}, {" E": -0.1}]
    return _choice(tokens, tops, text="A B C D E"), PROMPT, SENT, CAP_BONUS_UNAVAILABLE


def _ambiguous_bonus():
    ch = good_classic()
    ch["logprobs"]["top_logprobs"][5] = {" F": -0.1, " G": -0.1}   # tie at the bonus
    return ch, PROMPT, SENT, CAP_BONUS_UNAVAILABLE


def _both_shifts_match():
    # continuation " D D" verifies at BOTH shift 0 and shift 1.
    tokens = ["A", " B", " C", " D", " D"]
    tops = [None, {" B": -0.1}, {" D": -0.1}, {" D": -0.1}, {" D": -0.1}]
    return _choice(tokens, tops, text="A B C D D"), "A B C", "A B C D D", CAP_AMBIGUOUS_ALIGN


def _one_mismatch():
    # one continuation token is not the greedy argmax at any shift.
    ch = good_classic()
    ch["logprobs"]["top_logprobs"][4] = {" Z": -0.1}
    return ch, PROMPT, SENT, CAP_BAD_ALIGN


def _mismatched_lengths():
    ch = good_classic()
    ch["logprobs"]["top_logprobs"] = ch["logprobs"]["top_logprobs"][:5]   # 5 vs 6 tokens
    return ch, PROMPT, SENT, CAP_MALFORMED_ARRAYS


def _nan_token_logprob():
    ch = good_classic()
    ch["logprobs"]["token_logprobs"][2] = float("nan")
    return ch, PROMPT, SENT, CAP_MALFORMED_ARRAYS


def _nonnumeric_candidate():
    ch = good_classic()
    ch["logprobs"]["top_logprobs"][3] = {" D": "high"}   # not a number
    return ch, PROMPT, SENT, CAP_MALFORMED_ARRAYS


def _tops_not_dict():
    ch = good_classic()
    ch["logprobs"]["top_logprobs"][3] = "notadict"
    return ch, PROMPT, SENT, CAP_MALFORMED_ARRAYS


def _missing_shape():
    return {"text": "A B C D E F", "logprobs": {}}, PROMPT, SENT, CAP_BAD_SHAPE


def _byte_fallback_candidate():
    ch = good_classic()
    ch["logprobs"]["top_logprobs"][3] = {" D": -0.1, "<0xE2>": -0.2}
    return ch, PROMPT, SENT, CAP_UNSUPPORTED_TOKEN_IDENTITY


def _byte_offsets_multibyte():
    # café X with UTF-8 *byte* offsets (é is 2 bytes) — must be rejected.
    tokens = ["café", " X"]
    offsets = [0, 5]                       # byte offsets; code-point offsets are [0,4]
    tops = [None, {" X": -0.1}]
    ch = _choice(tokens, tops, text="café X", offsets=offsets, token_lps=[None, -0.1])
    return ch, "café", "café X", CAP_UNSUPPORTED_OFFSET_UNITS


ADVERSARIAL = [
    _echo_ignored, _echo_incomplete, _partial_coverage, _missing_continuation_position,
    _nonmonotonic_offsets, _overlapping_offsets, _gapped_offsets, _missing_bonus,
    _ambiguous_bonus, _both_shifts_match, _one_mismatch, _mismatched_lengths,
    _nan_token_logprob, _nonnumeric_candidate, _tops_not_dict, _missing_shape,
    _byte_fallback_candidate, _byte_offsets_multibyte,
]


@pytest.mark.parametrize("case", ADVERSARIAL, ids=lambda f: f.__name__.strip("_"))
def test_incompatible_endpoint_is_classified_unusable(case):
    choice, prompt, sent, expected = case()
    cap = classify_scored_choice(copy.deepcopy(choice), prompt, sent)
    assert cap.status == expected, f"{case.__name__}: got {cap.status} ({cap.detail})"
    assert not cap.usable


def test_byte_offsets_multibyte_rejected():
    choice, prompt, sent, expected = _byte_offsets_multibyte()
    cap = classify_scored_choice(choice, prompt, sent)
    assert cap.status == expected
    assert not cap.usable
    assert cap.offset_unit == "byte"


def test_codepoint_offsets_multibyte_ok():
    # The same café X but with code-point offsets is the accepted case.
    tokens = ["café", " X", " Y"]           # " Y" is the appended bonus token
    tops = [None, {" X": -0.1}, {" Y": -0.1}]
    ch = _choice(tokens, tops, text="café X Y", token_lps=[None, -0.1, -0.1])
    cap = classify_scored_choice(ch, "café", "café X")
    assert cap.usable
    assert cap.status == CAP_CLASSIC
    assert cap.offset_unit == "codepoint"


# --------------------------------------------------------------------------- #
# Generation policy: unknown/non-greedy fields are rejected.
# --------------------------------------------------------------------------- #

def test_policy_is_deterministic_accepts_greedy():
    ok, _ = policy_is_deterministic({"temperature": 0.0, "top_k": 1, "top_p": 1.0})
    assert ok


@pytest.mark.parametrize("policy", [
    {"temperature": 0.7},
    {"temperature": 0.0, "top_p": 0.9, "unknown_knob": 3},
    {"temperature": 0.0, "repeat_penalty": 1.1},
    {"temperature": 0.0, "mirostat": 2},
    {"temperature": 0.0, "grammar": "root ::= .*"},
    {"temperature": 0.0, "logit_bias": {"5": 100}},
])
def test_policy_is_deterministic_rejects_nongreedy(policy):
    ok, reason = policy_is_deterministic(policy)
    assert not ok and reason


def test_response_advertising_nongreedy_policy_is_rejected():
    cap = classify_scored_choice(good_classic(), PROMPT, SENT,
                                 obj={"generation_policy": {"temperature": 0.7}})
    assert cap.status == CAP_NONDETERMINISTIC_POLICY
    assert not cap.usable


def test_response_advertising_unknown_policy_field_is_rejected():
    cap = classify_scored_choice(good_classic(), PROMPT, SENT,
                                 obj={"sampling": {"temperature": 0.0, "mystery": 1}})
    assert cap.status == CAP_NONDETERMINISTIC_POLICY
    assert not cap.usable


# --------------------------------------------------------------------------- #
# End-to-end: an incompatible HTTP endpoint yields plain generation only.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def sim():
    servers = []

    def _make(**kw):
        eng = SimEngine(lm=LagLM(lag=6), **kw)
        server, base = start_sim_server(eng)
        servers.append(server)
        return base

    yield _make
    for s in servers:
        s.shutdown()
        s.server_close()


def test_byte_offset_endpoint_probes_unusable_and_falls_back_to_plain(sim):
    prompt = "café über naïve café über naïve café über"
    base = sim(offset_unit="byte")
    cap = probe_endpoint(base, "", "sim", probe_prompt=prompt)
    assert cap.status == CAP_UNSUPPORTED_OFFSET_UNITS
    assert not cap.usable
    # An unusable capability must produce plain-generation output only.
    expected = generate_burst(base, "", "sim", prompt, 40).text
    got, stats = spec_generate(base, "", "sim", prompt, max_tokens=40, capability=cap)
    assert got == expected
    assert stats.spec_available is False
    assert stats.verify_rounds == 0
