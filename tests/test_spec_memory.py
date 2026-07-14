"""LookupMemory: the draft-free, tokenizer-agnostic proposer."""

from __future__ import annotations

import pytest

from sclab.spec.memory import LookupMemory


def test_proposes_earlier_continuation_not_self_match():
    m = LookupMemory(shingle=8, positions_per_key=8)
    # "abcdefgh" was once followed by "XYZ..."; when we see it again the
    # proposal must be that continuation, not the empty self-match at the end.
    m.observe("abcdefghXYZ0123456789 tail padding here")
    m.observe(" and again abcdefgh")
    got = m.propose("and again abcdefgh", max_chars=8, min_chars=3)
    assert got is not None
    assert got.startswith("XYZ")


def test_returns_none_without_a_prior_occurrence():
    m = LookupMemory(shingle=8)
    m.observe("unique content with no repeats at all here")
    assert m.propose("...no matching suffix zzzz", max_chars=16, min_chars=4) is None


def test_returns_none_when_context_shorter_than_shingle():
    m = LookupMemory(shingle=16)
    m.observe("short")
    assert m.propose("short", max_chars=16) is None


def test_trims_to_whitespace_boundary():
    m = LookupMemory(shingle=8)
    m.observe("PREFIX##one two three four five six seven eight")
    m.observe(" repeat PREFIX##")
    got = m.propose("repeat PREFIX##", max_chars=12, min_chars=3)
    assert got is not None
    # 'one two three'[:12] = 'one two thre'; cut back to last space -> 'one two '
    assert got == "one two "


def test_most_recent_earlier_occurrence_wins():
    m = LookupMemory(shingle=6, positions_per_key=8)
    m.observe("KEYABCfirst_______ KEYABCsecond______ KEYABC")
    got = m.propose("...... KEYABC", max_chars=6, min_chars=3)
    assert got is not None
    # The most recent earlier 'KEYABC' was followed by 'second'.
    assert got.startswith("second")


def test_survives_buffer_trim():
    m = LookupMemory(shingle=8, max_bytes=200)
    # Establish a repeated pattern, then overflow to force a trim+rebuild.
    m.observe("PATTERNX-cont-inues-here-yep ")
    for _ in range(40):
        m.observe("filler text to grow the buffer past the cap ")
    m.observe("PATTERNX")
    # Should not raise and index must stay internally consistent.
    _ = m.propose("....... PATTERNX", max_chars=8, min_chars=2)
    assert len(m) <= 200


def test_shingle_must_be_reasonable():
    with pytest.raises(ValueError):
        LookupMemory(shingle=2)
