"""Opt-in integration tests against a real OpenAI-compatible server.

These are skipped unless ``SCLAB_SPEC_TEST_UPSTREAM`` (and optionally
``SCLAB_SPEC_TEST_MODEL``) point at a running endpoint, so normal CI stays fast
and weight-free. They were developed against a llama-cpp-python 0.3.16 server
hosting the synthetic ``tiny-cycle-gpt2`` fixture (a confident, canonical,
untrained model — see ``docs/spec_phase1_results.md``):

    python -m llama_cpp.server --model tiny-cycle-gpt2.gguf \\
        --n_ctx 8192 --logits_all true --port 8081
    SCLAB_SPEC_TEST_UPSTREAM=http://127.0.0.1:8081/v1 pytest tests/test_spec_realengine.py
"""

from __future__ import annotations

import os

import pytest

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.verify import generate_burst, probe_endpoint

UPSTREAM = os.environ.get("SCLAB_SPEC_TEST_UPSTREAM")
MODEL = os.environ.get("SCLAB_SPEC_TEST_MODEL", "m")

pytestmark = pytest.mark.skipif(
    not UPSTREAM, reason="set SCLAB_SPEC_TEST_UPSTREAM to run real-engine tests")


@pytest.fixture(scope="module")
def cap():
    c = probe_endpoint(UPSTREAM, "", MODEL)
    if not c.usable:
        pytest.skip(f"endpoint not usable for speculation: {c.status} ({c.detail})")
    return c


def test_probe_reports_a_measured_alignment(cap):
    assert cap.usable
    assert cap.shift in (0, 1)
    assert cap.echoed and cap.has_prompt_logprobs and cap.continuation_verified
    assert cap.offsets_ok


@pytest.mark.parametrize("prompt", [
    '{"name": "Ada", "role":',
    "def fibonacci(n):",
    "Step 1: first we compute",
    "The morning was cold and",
])
@pytest.mark.parametrize("max_tokens", [1, 16, 64, 200])
def test_spec_is_byte_identical_to_plain_generation(cap, prompt, max_tokens):
    expected = generate_burst(UPSTREAM, "", MODEL, prompt, max_tokens).text
    got, stats = spec_generate(UPSTREAM, "", MODEL, prompt, max_tokens=max_tokens,
                               shift=cap.shift or 0, draft_chars=96, burst_tokens=16)
    assert got == expected, "speculative output diverged from plain generation"
    assert stats.error is None
    assert stats.tokens_total <= max_tokens


def test_warm_memory_lands_real_drafts_and_stays_lossless(cap):
    prompt = "The report shows that"
    mem = LookupMemory()
    primer = generate_burst(UPSTREAM, "", MODEL, prompt, 260).text
    mem.observe(prompt + primer)
    got, stats = spec_generate(UPSTREAM, "", MODEL, prompt, max_tokens=256, memory=mem,
                               shift=cap.shift or 0, draft_chars=96, burst_tokens=16)
    assert got == generate_burst(UPSTREAM, "", MODEL, prompt, 256).text
    # Real accepted *draft* tokens (not just corrections) dominate the run.
    assert stats.tokens_accepted > 0
    assert stats.draft_tokens_accepted_per_verify > 1.0


def test_wrong_shift_would_diverge(cap):
    # The measured shift is load-bearing: the other shift must NOT be lossless,
    # which is exactly why the probe (not an assumption) picks it.
    prompt = "Step 1: first we compute"
    expected = generate_burst(UPSTREAM, "", MODEL, prompt, 200).text
    wrong = 1 - (cap.shift or 0)
    got, _ = spec_generate(UPSTREAM, "", MODEL, prompt, max_tokens=200,
                           shift=wrong, draft_chars=96, burst_tokens=16)
    assert got != expected
