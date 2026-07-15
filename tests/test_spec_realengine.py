"""Opt-in integration tests against real engines. Skipped unless configured.

Two independent opt-in lanes, neither of which downloads a model:

* **Text-surface mode** — set ``SCLAB_SPEC_TEST_UPSTREAM`` (and optionally
  ``SCLAB_SPEC_TEST_MODEL``) at a running OpenAI-compatible ``/v1`` server.
  Developed against a ``llama-cpp-python`` 0.3.16 server hosting the synthetic
  ``tiny-cycle-gpt2`` fixture (see ``docs/spec_phase1_results.md``)::

      python -m llama_cpp.server --model tiny-cycle-gpt2.gguf \\
          --n_ctx 8192 --logits_all true --port 8081
      SCLAB_SPEC_TEST_UPSTREAM=http://127.0.0.1:8081/v1 pytest tests/test_spec_realengine.py

* **Token-ID mode** — set ``SCLAB_SPEC_TEST_GGUF`` to an *already-present* local
  GGUF and (optionally) ``SCLAB_SPEC_TEST_MODEL_TYPE=trained|synthetic``. Uses the
  in-process ``llama_cpp.Llama`` API to verify drafts on token ids::

      SCLAB_SPEC_TEST_GGUF=/abs/path/model.gguf \\
      SCLAB_SPEC_TEST_MODEL_TYPE=trained pytest tests/test_spec_realengine.py

  When no GGUF is configured (or ``llama_cpp`` is not installed), the token-ID
  tests skip — no trained-model result is ever substituted by a synthetic one.
"""

from __future__ import annotations

import os

import pytest

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.token_verify import spec_generate_tokens
from sclab.spec.verify import generate_burst, probe_endpoint

# --------------------------------------------------------------------------- #
# Text-surface mode (HTTP endpoint).
# --------------------------------------------------------------------------- #

UPSTREAM = os.environ.get("SCLAB_SPEC_TEST_UPSTREAM")
MODEL = os.environ.get("SCLAB_SPEC_TEST_MODEL", "m")

text_mode = pytest.mark.skipif(
    not UPSTREAM, reason="set SCLAB_SPEC_TEST_UPSTREAM to run text-mode real-engine tests")


@pytest.fixture(scope="module")
def cap():
    c = probe_endpoint(UPSTREAM, "", MODEL)
    if not c.usable:
        pytest.skip(f"endpoint not usable for speculation: {c.status} ({c.detail})")
    return c


@text_mode
def test_probe_reports_a_measured_alignment(cap):
    assert cap.usable
    assert cap.shift in (0, 1)
    assert cap.echoed and cap.has_prompt_logprobs and cap.continuation_verified
    assert cap.offsets_ok and cap.bonus_ok
    assert cap.offset_unit == "codepoint"


@text_mode
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
                               capability=cap, draft_chars=96, burst_tokens=16)
    assert got == expected, "speculative output diverged from plain generation"
    assert stats.error is None
    assert stats.tokens_total <= max_tokens


@text_mode
def test_warm_memory_lands_real_drafts_and_stays_lossless(cap):
    prompt = "The report shows that"
    mem = LookupMemory()
    primer = generate_burst(UPSTREAM, "", MODEL, prompt, 260).text
    mem.observe(prompt + primer)
    got, stats = spec_generate(UPSTREAM, "", MODEL, prompt, max_tokens=256, memory=mem,
                               capability=cap, draft_chars=96, burst_tokens=16)
    assert got == generate_burst(UPSTREAM, "", MODEL, prompt, 256).text
    assert stats.tokens_accepted > 0
    assert stats.draft_tokens_accepted_per_verify > 1.0


@text_mode
def test_unusable_capability_falls_back_to_plain(cap):
    # A None capability must never speculate — plain generation only.
    prompt = "The morning was cold and"
    expected = generate_burst(UPSTREAM, "", MODEL, prompt, 64).text
    got, stats = spec_generate(UPSTREAM, "", MODEL, prompt, max_tokens=64, capability=None)
    assert got == expected
    assert stats.spec_available is False


# --------------------------------------------------------------------------- #
# Token-ID mode (in-process llama_cpp.Llama).
# --------------------------------------------------------------------------- #

GGUF = os.environ.get("SCLAB_SPEC_TEST_GGUF")
MODEL_TYPE = os.environ.get("SCLAB_SPEC_TEST_MODEL_TYPE", "synthetic")

token_mode = pytest.mark.skipif(
    not GGUF, reason="set SCLAB_SPEC_TEST_GGUF to an existing local GGUF for token-ID tests")


@pytest.fixture(scope="module")
def backend():
    if not GGUF:
        pytest.skip("no SCLAB_SPEC_TEST_GGUF configured")
    if not os.path.exists(GGUF):
        pytest.skip(f"GGUF not found at {GGUF}")
    try:
        from sclab.spec.llamacpp_backend import LlamaCppBackend
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"llama_cpp import failed: {exc}")
    b = LlamaCppBackend.from_model_path(GGUF, n_ctx=4096)
    cap = b.capability()
    if not cap.usable:
        pytest.skip(f"backend not usable for token-ID verification: {cap.detail}")
    return b


def _assert_token_and_byte_identical(backend, prompt, max_tokens, warm=False):
    prompt_ids = backend.encode_context(prompt)
    base = backend.generate_plain(prompt_ids, max_tokens)
    base_bytes = backend.decode_tokens(base.token_ids)
    mem = LookupMemory()
    if warm:
        mem.observe(prompt + base_bytes.decode("utf-8", errors="replace"))
    g = spec_generate_tokens(backend, prompt, max_tokens=max_tokens, memory=mem,
                             capability=backend.capability(), draft_chars=96, burst_tokens=16)
    assert g.token_ids == base.token_ids, "token-ID spec diverged from plain generation"
    assert g.text_bytes == base_bytes, "byte output diverged from plain generation"
    assert g.stats.tokens_total <= max_tokens
    return g


SYNTHETIC_PROMPTS = ["Step 1: first we compute", "the quick brown fox jumps over the lazy dog and then"]

TRAINED_PROMPTS = [
    '{"name": "Ada", "role": "engineer", "skills": [',      # JSON
    "You are a tool. Schema: {type: object, properties",    # repeated tool schema
    "def fibonacci(n):\n    if n < 2:\n        return n",    # code
    'The document states: "the balance was',                # quoted RAG text
    "The morning was cold and the streets were",            # free-form prose
    "café Shqipëria është e bukur — naïve façade",          # unicode + Albanian
]


@token_mode
@pytest.mark.parametrize("max_tokens", [1, 16, 64, 200])
def test_token_id_equality_synthetic(backend, max_tokens):
    for prompt in SYNTHETIC_PROMPTS:
        _assert_token_and_byte_identical(backend, prompt, max_tokens)


@token_mode
def test_token_id_equality_with_warm_memory(backend):
    _assert_token_and_byte_identical(backend, SYNTHETIC_PROMPTS[1], 200, warm=True)


@token_mode
@pytest.mark.skipif(MODEL_TYPE != "trained",
                    reason="set SCLAB_SPEC_TEST_MODEL_TYPE=trained with a trained GGUF")
@pytest.mark.parametrize("prompt", TRAINED_PROMPTS)
@pytest.mark.parametrize("max_tokens", [16, 64, 200])
def test_token_id_equality_trained_workloads(backend, prompt, max_tokens):
    # Token-ID + byte equality across JSON, tool schema, code, quoted RAG, prose,
    # and Unicode/Albanian on a real trained model — the evidence Phase 1 lacked.
    _assert_token_and_byte_identical(backend, prompt, max_tokens, warm=True)
