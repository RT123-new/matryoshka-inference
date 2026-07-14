"""End-to-end alignment + capability tests over real HTTP against the sim.

The sim can emulate both logprob conventions (classic and the +1-shifted one
llama-cpp-python actually uses). These tests prove the loop is lossless *only*
when told the correct shift, that the probe detects the shift by behaviour, and
that unusable endpoints (echo ignored) are refused rather than mis-verified.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sclab.spec.bench import run_bench
from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.sim import LagLM, SimEngine, start_sim_server
from sclab.spec.verify import (
    CAP_BAD_SHAPE,
    CAP_CLASSIC,
    CAP_ECHO_IGNORED,
    CAP_SHIFTED,
    generate_burst,
    probe_endpoint,
)

PROMPT = "the quick brown fox jumps over the lazy dog and then the"


@pytest.fixture()
def sim():
    servers = []

    def _make(lag=10, shift=0, max_total=100_000):
        eng = SimEngine(lm=LagLM(lag=lag), max_total_tokens=max_total, logprob_shift=shift)
        server, base = start_sim_server(eng)
        servers.append(server)
        return base

    yield _make
    for s in servers:
        s.shutdown()
        s.server_close()


@pytest.mark.parametrize("shift", [0, 1])
def test_probe_detects_alignment(sim, shift):
    base = sim(shift=shift)
    cap = probe_endpoint(base, "", "sim")
    assert cap.usable
    assert cap.shift == shift
    assert cap.status == (CAP_CLASSIC if shift == 0 else CAP_SHIFTED)
    assert cap.echoed and cap.has_prompt_logprobs and cap.continuation_verified


@pytest.mark.parametrize("shift", [0, 1])
def test_lossless_only_at_correct_shift(sim, shift):
    base = sim(shift=shift)
    expected = generate_burst(base, "", "sim", PROMPT, 80).text
    got, stats = spec_generate(base, "", "sim", PROMPT, max_tokens=80,
                               shift=shift, draft_chars=96, burst_tokens=8)
    assert got == expected
    assert stats.tokens_accepted > 0          # real draft tokens, not just corrections
    # The wrong shift must diverge — this is the guard that pins the whole fix.
    wrong, _ = spec_generate(base, "", "sim", PROMPT, max_tokens=80,
                             shift=1 - shift, draft_chars=96, burst_tokens=8)
    assert wrong != expected


def test_run_bench_probes_and_reports_capability(sim):
    base = sim(shift=1)
    res = run_bench(base, "", "sim", PROMPT, max_tokens=80, draft_chars=96, burst_tokens=8)
    assert res["spec_available"] is True
    assert res["capability"] == CAP_SHIFTED
    assert res["shift"] == 1
    assert res["identical_output"] is True


def test_telemetry_separates_accepted_drafts_from_corrections(sim):
    base = sim(shift=0)
    mem = LookupMemory()
    primer = generate_burst(base, "", "sim", PROMPT, 120).text
    mem.observe(PROMPT + primer)
    _, stats = spec_generate(base, "", "sim", PROMPT, max_tokens=100, memory=mem,
                             shift=0, draft_chars=128, burst_tokens=8)
    s = stats.summary()
    # accepted-per-verify counts *draft* tokens only, and equals the raw ratio.
    assert stats.tokens_accepted > 0
    assert s["draft_tokens_accepted_per_verify"] == round(
        stats.tokens_accepted / stats.verify_rounds, 3)
    # emitted-per-verify additionally folds in corrections + bonus, so it is >=.
    assert s["tokens_emitted_per_verify"] >= s["draft_tokens_accepted_per_verify"]
    assert "corrections_per_verify" in s and "bonus_tokens_per_verify" in s


# --- an endpoint that ignores echo (like the native llama.cpp server) ------ #

class _EchoIgnoringHandler(BaseHTTPRequestHandler):
    """Returns logprobs for generated tokens only, never echoing the prompt."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        gen = " one two three"
        # classic-shaped logprobs, but only for the generated tail (no echo).
        body = json.dumps({
            "choices": [{
                "index": 0, "text": gen, "finish_reason": "length",
                "logprobs": {"tokens": [" one", " two", " three"],
                             "text_offset": [0, 4, 8],
                             "token_logprobs": [-0.1, -0.1, -0.1],
                             "top_logprobs": [{" one": -0.1}, {" two": -0.1}, {" three": -0.1}]},
            }],
            "usage": {"completion_tokens": 3},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_probe_refuses_echo_ignoring_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoIgnoringHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        cap = probe_endpoint(base, "", "m")
        assert not cap.usable
        assert cap.status in (CAP_ECHO_IGNORED, CAP_BAD_SHAPE)
        # run_bench must fall back to baseline-only, never a mis-verified spec run.
        res = run_bench(base, "", "m", "hello", max_tokens=8)
        assert res["spec_available"] is False
        assert "spec" not in res
    finally:
        server.shutdown()
        server.server_close()
