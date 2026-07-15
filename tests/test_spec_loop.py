"""The text-surface lossless-speculation invariant, proven end-to-end over HTTP.

A deterministic sim engine (exact echo+logprobs semantics) stands in for a real
OpenAI-compatible completions engine, so we can assert the loop's output is
*byte-identical* to a single plain generation call — the promise of the
approach — without weights or a GPU. Text mode proves *surface* identity only;
``test_spec_token_verify.py`` covers unconditional token-id equivalence.

Since PR-hardening, ``spec_generate`` refuses to speculate without a usable
capability, so these tests pass the sim's measured (classic, shift 0) capability
explicitly. Without one the loop is plain generation — a property tested in
``test_spec_capability_strict.py``.
"""

from __future__ import annotations

import pytest

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.sim import LagLM, SimEngine, start_sim_server
from sclab.spec.verify import CAP_CLASSIC, EndpointCapability, generate_burst

# The sim defaults to the classic (shift 0) convention; construct its usable
# capability directly so the loop enters the verify lane.
CLASSIC_CAP = EndpointCapability(
    status=CAP_CLASSIC, shift=0, echoed=True, has_prompt_logprobs=True,
    offsets_ok=True, offset_unit="codepoint", continuation_verified=True, bonus_ok=True)


@pytest.fixture()
def sim():
    servers = []

    def _make(lag=10, overhead_ms=0.0, prefill_ms=0.0, decode_ms=0.0, max_total=100_000):
        eng = SimEngine(lm=LagLM(lag=lag), overhead_ms=overhead_ms,
                        prefill_ms_per_token=prefill_ms, decode_ms_per_token=decode_ms,
                        max_total_tokens=max_total)
        server, base = start_sim_server(eng)
        servers.append(server)
        return base

    yield _make
    for s in servers:
        s.shutdown()
        s.server_close()


def _baseline(base, prompt, max_tokens):
    r = generate_burst(base, "", "sim", prompt, max_tokens)
    assert not r.error
    return r.text


REPETITIVE = "the quick brown fox jumps over the lazy dog and then the"
JSONISH = '{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name":'
PROSE = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"


@pytest.mark.parametrize("prompt", [REPETITIVE, JSONISH, PROSE])
@pytest.mark.parametrize("lag", [6, 10, 17])
@pytest.mark.parametrize("max_tokens", [1, 5, 32, 100])
def test_spec_output_is_byte_identical_to_plain_generation(sim, prompt, lag, max_tokens):
    base = sim(lag=lag)
    expected = _baseline(base, prompt, max_tokens)
    got, stats = spec_generate(base, "", "sim", prompt, max_tokens=max_tokens,
                               capability=CLASSIC_CAP, draft_chars=96, burst_tokens=8)
    assert got == expected
    assert stats.tokens_total <= max_tokens
    assert stats.error is None


def test_never_exceeds_token_budget(sim):
    base = sim(lag=8)
    for max_tokens in (1, 2, 3, 7, 13, 50):
        got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=max_tokens,
                                   capability=CLASSIC_CAP, draft_chars=128, burst_tokens=4)
        assert stats.tokens_total <= max_tokens
        assert got == _baseline(base, REPETITIVE, max_tokens)


def test_warm_memory_lands_drafts_and_stays_lossless(sim):
    base = sim(lag=10)
    # Prime the memory with the exact continuation, as an agent's repeated
    # tool schema or quoted context would. Output must still match plain decode.
    mem = LookupMemory()
    primer = _baseline(base, REPETITIVE, 120)
    mem.observe(REPETITIVE + primer)
    got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=100,
                               capability=CLASSIC_CAP, memory=mem, draft_chars=128, burst_tokens=8)
    assert got == _baseline(base, REPETITIVE, 100)
    # With a warm memory the run should be dominated by accepted drafts, not bursts.
    assert stats.tokens_accepted > stats.tokens_burst


def test_persistent_memory_stays_lossless_and_high_acceptance(sim):
    base = sim(lag=10)
    mem = LookupMemory()
    got1, first = spec_generate(base, "", "sim", REPETITIVE, max_tokens=80, memory=mem,
                                capability=CLASSIC_CAP, draft_chars=128, burst_tokens=8)
    got2, second = spec_generate(base, "", "sim", REPETITIVE, max_tokens=80, memory=mem,
                                 capability=CLASSIC_CAP, draft_chars=128, burst_tokens=8)
    # Reusing one memory across two requests stays lossless on both...
    assert got1 == got2 == _baseline(base, REPETITIVE, 80)
    # ...and both are dominated by verified drafts, not sequential bursts.
    assert second.accepted_per_verify > 5.0
    assert second.tokens_accepted > second.tokens_burst


def test_seam_fallback_does_not_trigger_long_backoff(sim):
    # A tokenization seam is a one-token hiccup: it must cost at most a short
    # burst, never the multi-round acceptance-collapse backoff.
    base = sim(lag=10)
    mem = LookupMemory()
    primer = _baseline(base, REPETITIVE, 200)
    mem.observe(REPETITIVE + primer)
    _, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=150, memory=mem,
                             capability=CLASSIC_CAP, draft_chars=200, burst_tokens=8)
    # Even if a few seams occur, bursts stay a small fraction of total tokens.
    assert stats.tokens_burst <= 8 * (stats.seam_fallbacks + 1)


def test_lossless_when_generation_stops_early(sim):
    # max_total_tokens forces the engine to stop before max_tokens; the loop
    # must stop at exactly the same place as a plain call.
    base = sim(lag=6, max_total=40)  # sim stops once total tokens hit 40
    expected = _baseline(base, REPETITIVE, 100)
    got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=100,
                               capability=CLASSIC_CAP, draft_chars=96, burst_tokens=8)
    assert got == expected
    assert stats.finish_reason == "stop"


def test_unusable_capability_falls_back_to_plain(sim):
    # No capability (or an unusable one) => plain generation only, still lossless.
    base = sim(lag=10)
    expected = _baseline(base, REPETITIVE, 60)
    got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=60, capability=None)
    assert got == expected
    assert stats.spec_available is False
    assert stats.verify_rounds == 0
    assert stats.tokens_total <= 60


UNICODE_PROMPTS = [
    "plain ascii repeated plain ascii repeated plain ascii",
    "café au lait café au lait café au lait café au lait",
    "Shqipëria është e bukur Shqipëria është e bukur Shqipëria",
    "😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀",
    "“curly” — em–dash “curly” — em–dash “curly” — em–dash",
    "é à ô é à ô é à ô é à ô é à ô é à ô",
    "line\n\n\nbreaks line\n\n\nbreaks line\n\n\nbreaks line",
    "punct!!! ??? ... punct!!! ??? ... punct!!! ??? ...",
]


@pytest.mark.parametrize("prompt", UNICODE_PROMPTS, ids=lambda p: repr(p[:14]))
@pytest.mark.parametrize("max_tokens", [16, 48])
def test_unicode_and_seam_byte_identity_text_mode(sim, prompt, max_tokens):
    # Code-point-offset endpoint (the accepted case): the loop must stay
    # byte-identical to plain generation across multibyte text and whitespace
    # seams. (Byte-offset endpoints are rejected — see test_spec_capability_strict.)
    base = sim(lag=6)
    mem = LookupMemory()
    mem.observe(prompt + _baseline(base, prompt, 160))
    got, stats = spec_generate(base, "", "sim", prompt, max_tokens=max_tokens,
                               capability=CLASSIC_CAP, memory=mem, draft_chars=64, burst_tokens=8)
    assert got == _baseline(base, prompt, max_tokens)
    assert stats.tokens_total <= max_tokens


def test_fewer_engine_roundtrips_on_repetitive_workload(sim):
    # The mechanism that produces the speedup is deterministic: many tokens
    # verified per engine round-trip instead of one token per sequential decode
    # step. That ratio — not wall-clock, which depends on the host's per-request
    # overhead and is what `sclab spec-bench` measures — is what we assert here.
    base = sim(lag=10)
    from sclab.spec.bench import run_bench

    res = run_bench(base, "", "sim", REPETITIVE, max_tokens=120,
                    draft_chars=96, burst_tokens=8)
    assert res["identical_output"] is True
    # ~10 tokens per round-trip => ~10x fewer sequential decode steps, which is
    # the whole win on any decode-bound engine.
    assert res["spec"]["tokens_per_request"] > 3.0
    assert res["spec"]["accepted_per_verify"] > 5.0
