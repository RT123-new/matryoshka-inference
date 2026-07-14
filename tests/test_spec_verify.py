"""Unit tests for the scoring adapter: positional alignment, conservative
acceptance, the bonus position, offsets, and the behavioural capability probe.

These are pure and deterministic — they build ``choice`` dicts by hand and never
touch a network — so they pin the *semantics* the loop depends on. A separate,
opt-in module exercises a real llama-cpp-python server.
"""

from __future__ import annotations

import json
from pathlib import Path

from sclab.spec.verify import (
    Prediction,
    ScoreResult,
    _parse_logprobs,
    _prediction_from_top,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _choice(tokens, offsets, token_lps, tops, text=None):
    return {
        "text": text if text is not None else "".join(tokens),
        "logprobs": {"tokens": tokens, "text_offset": offsets,
                     "token_logprobs": token_lps, "top_logprobs": tops},
    }


# --- prediction reduction (conservative acceptance) ------------------------ #

def test_prediction_unique_top_is_usable():
    p = _prediction_from_top({" B": -0.1, " C": -3.0})
    assert p == Prediction(surface=" B", logprob=-0.1, ambiguous=False)


def test_prediction_top_tie_is_ambiguous():
    # Two candidates share the top logprob: greedy tie-break is by token id,
    # which text scoring cannot see, so we must not accept.
    p = _prediction_from_top({" B": -0.1, " C": -0.1})
    assert p.ambiguous is True


def test_prediction_empty_surface_is_ambiguous():
    p = _prediction_from_top({"": -0.1, " B": -2.0})
    assert p.ambiguous is True  # a special/empty surface cannot be appended as text


def test_prediction_missing_top_is_ambiguous():
    assert _prediction_from_top(None).ambiguous is True
    assert _prediction_from_top({}).ambiguous is True


def test_is_greedy_requires_unique_exact_match():
    toks, _ = _parse_logprobs(_choice(
        tokens=["A", " B"], offsets=[0, 1], token_lps=[None, -0.1],
        tops=[None, {" B": -0.1, " C": -3.0}]), shift=0)
    assert toks[1].is_greedy is True
    # A tie at the top must not be accepted even if our surface is the argmax.
    toks2, _ = _parse_logprobs(_choice(
        tokens=["A", " B"], offsets=[0, 1], token_lps=[None, -0.1],
        tops=[None, {" B": -0.1, " C": -0.1}]), shift=0)
    assert toks2[1].is_greedy is False


# --- positional alignment (classic vs shifted) ----------------------------- #

CLASSIC = _choice(
    tokens=["A", " B", " C", " D", " E"], offsets=[0, 1, 3, 5, 7],
    token_lps=[None, -0.1, -0.1, -0.1, -0.1],
    # classic: top_logprobs[i] predicts token i
    tops=[None, {" B": -0.1}, {" C": -0.1}, {" D": -0.1}, {" E": -0.1}])

SHIFTED = _choice(
    tokens=["A", " B", " C", " D"], offsets=[0, 1, 3, 5],
    token_lps=[None, -9.0, -9.0, -9.0],
    # shifted: top_logprobs[i] predicts token i+1 (last one predicts the bonus)
    tops=[{" B": -0.1}, {" C": -0.1}, {" D": -0.1}, {" E": -0.1}])


def test_classic_alignment_verifies_greedy_continuation():
    toks, _ = _parse_logprobs(CLASSIC, shift=0)
    # tokens B, C, D are the greedy continuation and must all verify
    assert [t.is_greedy for t in toks] == [False, True, True, True, True]


def test_shifted_alignment_verifies_only_at_shift_1():
    good0, _ = _parse_logprobs(SHIFTED, shift=0)
    good1, _ = _parse_logprobs(SHIFTED, shift=1)
    # At shift 0 the shifted response mis-verifies (predicts the *next* token);
    # at shift 1 the greedy continuation verifies.
    assert [t.is_greedy for t in good0] == [False, False, False, False]
    assert [t.is_greedy for t in good1] == [False, True, True, True]


def test_greedy_after_bonus_classic_and_shifted():
    # sent = "A B C D" (len 7). Classic appends the generated token ' E'; shifted
    # carries the bonus in the last prediction. Both must yield ' E'.
    toks_c, preds_c = _parse_logprobs(CLASSIC, shift=0)
    sr_c = ScoreResult(tokens=toks_c, shift=0, predictions=preds_c)
    assert sr_c.greedy_after(7).surface == " E"

    toks_s, preds_s = _parse_logprobs(SHIFTED, shift=1)
    sr_s = ScoreResult(tokens=toks_s, shift=1, predictions=preds_s)
    assert sr_s.greedy_after(7).surface == " E"


def test_draft_tokens_and_seam_detection():
    toks, preds = _parse_logprobs(CLASSIC, shift=0)
    sr = ScoreResult(tokens=toks, shift=0, predictions=preds)
    # context = "A B" (len 3), draft = " C D" -> draft tokens are C and D.
    picked = sr.draft_tokens(3, 7)
    assert [t.surface for t in picked] == [" C", " D"]
    # A boundary that falls inside a token (straddles the seam) -> None.
    assert sr.draft_tokens(2, 7) is None       # cuts inside ' B' (offset 1..3)


# --- offsets, including multibyte text ------------------------------------- #

def test_offsets_are_trusted_in_string_units_multibyte():
    # llama-cpp-python reports text_offset in Python code-point units, matching
    # len(surface). Verify the parser's boundary math stays consistent when the
    # text contains multibyte characters (é is 1 code point, 2 UTF-8 bytes).
    text = "café X"
    tokens = ["café", " X"]
    offsets = [0, 4]        # code-point offsets, NOT byte offsets (which would be 0,5)
    toks, preds = _parse_logprobs(_choice(
        tokens=tokens, offsets=offsets, token_lps=[None, -0.1],
        tops=[None, {" X": -0.1}], text=text), shift=0)
    sr = ScoreResult(tokens=toks, shift=0, predictions=preds)
    # draft region covering just " X" must be picked cleanly past the é.
    picked = sr.draft_tokens(len("café"), len(text))
    assert [t.surface for t in picked] == [" X"]


# --- real-engine fixture regression (no server needed) --------------------- #

def test_real_llamacpp_fixture_is_shifted_by_one():
    """A verbatim llama-cpp-python 0.3.16 echo response: its greedy continuation
    verifies at shift 1, not shift 0. Guards against a silent alignment regression.
    """
    fx = FIXTURES / "llamacpp_python_echo_shift.json"
    choice = json.loads(fx.read_text())["choices"][0]
    sent = "Step 1: first we compute they int out who their more one"
    prompt = "Step 1: first we compute"

    def match_rate(shift):
        toks, _ = _parse_logprobs(choice, shift=shift)
        good = total = 0
        for t in toks:
            if len(prompt) <= t.offset and t.offset + len(t.surface) <= len(sent):
                total += 1
                good += t.is_greedy
        return good, total

    g0, total = match_rate(0)
    g1, _ = match_rate(1)
    assert total >= 6
    assert g1 == total           # every continuation token verifies at shift 1
    assert g0 == 0               # and none of them at shift 0
