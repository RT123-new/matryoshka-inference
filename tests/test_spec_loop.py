"""The lossless-speculation invariant, proven end-to-end over real HTTP.

A deterministic sim engine (exact echo+logprobs semantics) stands in for a
real OpenAI-compatible completions engine, so we can assert the loop's output
is *byte-identical* to a single plain generation call — the whole promise of
the approach — without weights or a GPU.
"""

from __future__ import annotations

import pytest

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.sim import LagLM, SimEngine, start_sim_server
from sclab.spec.verify import generate_burst


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
                               draft_chars=96, burst_tokens=8)
    assert got == expected
    assert stats.tokens_total <= max_tokens
    assert stats.error is None


def test_never_exceeds_token_budget(sim):
    base = sim(lag=8)
    for max_tokens in (1, 2, 3, 7, 13, 50):
        got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=max_tokens,
                                   draft_chars=128, burst_tokens=4)
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
                               memory=mem, draft_chars=128, burst_tokens=8)
    assert got == _baseline(base, REPETITIVE, 100)
    # With a warm memory the run should be dominated by accepted drafts, not bursts.
    assert stats.tokens_accepted > stats.tokens_burst


def test_persistent_memory_stays_lossless_and_high_acceptance(sim):
    base = sim(lag=10)
    mem = LookupMemory()
    got1, first = spec_generate(base, "", "sim", REPETITIVE, max_tokens=80, memory=mem,
                                draft_chars=128, burst_tokens=8)
    got2, second = spec_generate(base, "", "sim", REPETITIVE, max_tokens=80, memory=mem,
                                 draft_chars=128, burst_tokens=8)
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
                             draft_chars=200, burst_tokens=8)
    # Even if a few seams occur, bursts stay a small fraction of total tokens.
    assert stats.tokens_burst <= 8 * (stats.seam_fallbacks + 1)


def test_lossless_when_generation_stops_early(sim):
    # max_total_tokens forces the engine to stop before max_tokens; the loop
    # must stop at exactly the same place as a plain call.
    base = sim(lag=6, max_total=40)  # sim stops once total tokens hit 40
    expected = _baseline(base, REPETITIVE, 100)
    got, stats = spec_generate(base, "", "sim", REPETITIVE, max_tokens=100,
                               draft_chars=96, burst_tokens=8)
    assert got == expected
    assert stats.finish_reason == "stop"


def test_reports_speedup_on_decode_bound_engine(sim):
    # With sequential decode priced 10x the parallel prefill, verifying many
    # tokens per round must beat one-token-per-step decoding on wall clock.
    base = sim(lag=10, overhead_ms=2, prefill_ms=1, decode_ms=10)
    from sclab.spec.bench import run_bench

    res = run_bench(base, "", "sim", REPETITIVE, max_tokens=120,
                    draft_chars=96, burst_tokens=8)
    assert res["identical_output"] is True
    assert res["speedup"] > 1.5
    assert res["spec"]["tokens_per_request"] > 2.0
