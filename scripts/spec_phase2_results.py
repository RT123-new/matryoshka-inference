"""Regenerate ``results/spec_phase2/`` from deterministic fixtures.

Everything here is **synthetic** — a deterministic sim engine for the
text-surface lane and a deterministic fake token backend for the token-ID lane.
No real or trained model is loaded (none was reachable in the build
environment). The point is a reproducible, machine-readable demonstration of:

* the strict text-mode capability probe accepting a code-point endpoint and
  rejecting a byte-offset one,
* the text-surface loop being byte-identical to plain generation,
* the token-ID loop being **id- and byte-identical** to plain generation,
* honest, correctness-gated timing (which is a *slowdown* on this rig).

Run: ``PYTHONPATH=src python scripts/spec_phase2_results.py``
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

from sclab.spec.backend import (
    DETERMINISTIC_POLICY,
    TOKEN_ID_VERIFIED,
    DraftVerification,
    GenerationResult,
    VerificationCapability,
)
from sclab.spec.bench import run_bench, run_timed_bench
from sclab.spec.memory import LookupMemory
from sclab.spec.sim import LagLM, SimEngine, start_sim_server
from sclab.spec.token_verify import spec_generate_tokens
from sclab.spec.verify import probe_endpoint

OUT = Path(__file__).resolve().parents[1] / "results" / "spec_phase2"
PROMPT = "the quick brown fox jumps over the lazy dog and then the"


# --- a tiny deterministic fake token backend (canonical char tokenizer) ----- #
class _Tok:
    BOS, EOS = 1, 2

    def __init__(self):
        self.id2b = {self.BOS: b"", self.EOS: b""}
        self.s2id = {}
        n = 10
        for c in [chr(x) for x in range(32, 127)]:
            self.id2b[n] = c.encode()
            self.s2id[c] = n
            n += 1

    def encode(self, text):
        return [self.BOS] + [self.s2id[c] for c in text]

    def decode(self, ids):
        return b"".join(self.id2b[i] for i in ids)


class _FakeTokenBackend:
    def __init__(self, lag=8):
        self.tok = _Tok()
        self.lag = lag
        self.seed = self.tok.s2id["x"]

    def capability(self):
        return VerificationCapability(TOKEN_ID_VERIFIED, deterministic=True, supports_bonus=True,
                                      eos_token_id=self.tok.EOS, policy=dict(DETERMINISTIC_POLICY))

    def encode_context(self, text):
        return self.tok.encode(text)

    def decode_tokens(self, ids):
        return self.tok.decode(ids)

    def _argmax(self, seq):
        body = [i for i in seq if i not in (self.tok.BOS, self.tok.EOS)]
        return body[-self.lag] if len(body) >= self.lag else self.seed

    def generate_plain(self, ctx, max_tokens):
        seq, out = list(ctx), []
        for _ in range(max_tokens):
            nid = self._argmax(seq)
            if nid == self.tok.EOS:
                return GenerationResult(out, "stop")
            out.append(nid)
            seq.append(nid)
        return GenerationResult(out, "length")

    def verify_draft(self, ctx, draft):
        return DraftVerification([self._argmax(list(ctx) + list(draft[:j])) for j in range(len(draft) + 1)])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    (OUT / "raw_responses").mkdir(parents=True, exist_ok=True)

    environment = {
        "note": "SYNTHETIC ONLY — deterministic sim engine + fake token backend. "
                "No real or trained model was loaded (none reachable in this environment).",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "text_mode_engine": "sclab.spec.sim.SimEngine (LagLM, synthetic)",
        "token_mode_backend": "scripts fake token backend (deterministic, synthetic)",
        "real_engine_tested": False,
        "trained_model_tested": False,
    }

    capability: dict = {}
    correctness: dict = {}
    raw: dict[str, str] = {}

    # -- text-surface lane: probe (codepoint usable, byte rejected) + gate ---- #
    # Overhead-dominated on purpose (no modeled decode cost): the sim's own
    # HTTP/Python per-request cost dominates, so speculation — which makes MORE
    # round-trips — is *slower*, exactly the honest CPU-rig result. A modeled
    # decode cost could manufacture a "speedup", which would prove nothing.
    cp = SimEngine(lm=LagLM(lag=10), offset_unit="codepoint")
    srv, base = start_sim_server(cp)
    try:
        cap = probe_endpoint(base, "", "sim")
        capability["text_codepoint"] = {"status": cap.status, "usable": cap.usable, "shift": cap.shift,
                                        "offset_unit": cap.offset_unit, "detail": cap.detail}
        bench = run_bench(base, "", "sim", PROMPT, max_tokens=80, capability=cap)
        correctness["text_surface_byte_identical"] = bench["identical_output"]
        raw["text_baseline"] = bench.get("baseline_text", "")
        raw["text_spec"] = bench.get("spec_text", "")
        timed = run_timed_bench(base, "", "sim", PROMPT, cap, max_tokens=80, samples=5)
    finally:
        srv.shutdown()
        srv.server_close()

    byte_eng = SimEngine(lm=LagLM(lag=10), offset_unit="byte")
    srv, base = start_sim_server(byte_eng)
    try:
        bcap = probe_endpoint(base, "", "sim")
        capability["text_byte_offsets"] = {"status": bcap.status, "usable": bcap.usable,
                                           "detail": bcap.detail}
    finally:
        srv.shutdown()
        srv.server_close()

    # -- token-ID lane: id + byte identity ----------------------------------- #
    backend = _FakeTokenBackend(lag=8)
    cap_t = backend.capability()
    capability["token_id"] = {"mode": cap_t.mode, "usable": cap_t.usable,
                              "deterministic": cap_t.deterministic, "policy": cap_t.policy}
    token_rows = []
    for mt in (1, 16, 64, 200):
        pids = backend.encode_context(PROMPT)
        base_gen = backend.generate_plain(pids, mt)
        base_bytes = backend.decode_tokens(base_gen.token_ids)
        mem = LookupMemory()
        mem.observe(PROMPT + backend.decode_tokens(backend.generate_plain(pids, 200).token_ids).decode())
        g = spec_generate_tokens(backend, PROMPT, max_tokens=mt, memory=mem, capability=cap_t,
                                 draft_chars=96, burst_tokens=8)
        token_rows.append({
            "max_tokens": mt,
            "token_id_equal": g.token_ids == base_gen.token_ids,
            "byte_equal": g.text_bytes == base_bytes,
            "verify_rounds": g.stats.token_id_verify_rounds,
            "draft_ids_accepted": g.stats.draft_ids_accepted,
            "draft_ids_accepted_per_verify": round(g.stats.draft_ids_accepted_per_verify, 3),
        })
        if mt == 200:
            raw["token_baseline"] = base_bytes.decode()
            raw["token_spec"] = g.text
    correctness["token_id_rows"] = token_rows
    correctness["token_id_all_equal"] = all(r["token_id_equal"] and r["byte_equal"] for r in token_rows)

    (OUT / "environment.json").write_text(json.dumps(environment, indent=2))
    (OUT / "capability.json").write_text(json.dumps(capability, indent=2, ensure_ascii=False))
    (OUT / "correctness.json").write_text(json.dumps(correctness, indent=2))
    (OUT / "benchmark.json").write_text(json.dumps(timed, indent=2))
    for k, v in raw.items():
        (OUT / "raw_responses" / f"{k}.txt").write_text(v)

    print(f"wrote {OUT}")
    print("token-ID all equal:", correctness["token_id_all_equal"])
    print("text byte-identical:", correctness["text_surface_byte_identical"])
    print("byte-offset endpoint rejected:", not capability["text_byte_offsets"]["usable"])


if __name__ == "__main__":
    main()
