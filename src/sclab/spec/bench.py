"""Honest measurement of API-level speculation against any engine.

Three separable measurements, in strict dependency order — **correctness gates
speed**:

* ``run_bench`` — the correctness gate plus a first-order round-trip ratio. Runs
  the same greedy request as one plain call and through the speculation loop and
  reports whether the outputs are **byte-identical**. A speed number is only
  populated when they are; a divergent run reports ``speedup=None`` and
  ``correctness_gate_passed=False``, never a speed claim.
* ``run_timed_bench`` — rigorous wall-clock, only meaningful once correctness
  passes. Alternates baseline/spec execution order, takes several samples,
  validates every timed call (a fast error is never counted as a fast success),
  separates cold-prefix / warm-prefix / fully-cached regimes, reports median with
  min/max, and records HTTP request count separately from wall time plus whether
  prefix caching was actually observed.
* ``run_cost_probe`` — content-independent engine physics: scoring *k* tokens
  (one parallel prefill) vs decoding *k* tokens sequentially.

``save_bench_results`` writes machine-readable JSON (and captures the raw texts),
so a published number can always be traced back to the run that produced it.

None of this fabricates a bandwidth-bound decoder; on an overhead-dominated CPU
rig speculation is *slower*, and the numbers say so (see
``docs/spec_phase1_results.md``).
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.verify import EndpointCapability, generate_burst, probe_endpoint, score_completion


def run_bench(upstream: str, api_key: str, model: str, prompt: str,
              max_tokens: int = 256, draft_chars: int = 64, burst_tokens: int = 16,
              warm_text: str | None = None, capability: EndpointCapability | None = None,
              timeout: int = 600) -> dict[str, Any]:
    """Correctness gate (byte-identity) + a first-order round-trip ratio.

    Probes the endpoint first (unless a ``capability`` is supplied) and only
    speculates when it is proven usable; against an unusable endpoint it reports
    the baseline and the reason, never a mis-verified speculative result. A speed
    number is reported **only** when the speculative output is byte-identical to
    the plain baseline.
    """
    if capability is None:
        capability = probe_endpoint(upstream, api_key, model, timeout=timeout)

    t0 = time.perf_counter()
    base = generate_burst(upstream, api_key, model, prompt, max_tokens, timeout=timeout)
    base_s = time.perf_counter() - t0
    if base.error:
        return {"error": f"baseline failed: {base.error}"}
    base_tokens = int((base.usage or {}).get("completion_tokens") or 0)
    baseline = {"seconds": round(base_s, 4), "tokens": base_tokens,
                "tok_s": round(base_tokens / base_s, 2) if base_s else None}

    if not capability.usable:
        return {
            "capability": capability.status,
            "capability_detail": capability.detail,
            "spec_available": False,
            "baseline": baseline,
            "note": f"endpoint cannot verify drafts ({capability.status}); "
                    "speculation disabled, plain generation only.",
        }

    memory = LookupMemory()
    if warm_text:
        memory.observe(warm_text)
    t0 = time.perf_counter()
    spec_text, stats = spec_generate(
        upstream, api_key, model, prompt,
        max_tokens=max_tokens, memory=memory, capability=capability,
        draft_chars=draft_chars, burst_tokens=burst_tokens, timeout=timeout,
    )
    spec_s = time.perf_counter() - t0

    identical = spec_text == base.text
    return {
        "capability": capability.status,
        "shift": capability.shift,
        "spec_available": True,
        "correctness_gate_passed": identical,
        "baseline": baseline,
        "spec": {"seconds": round(spec_s, 4), **stats.summary(),
                 "tok_s": round(stats.tokens_total / spec_s, 2) if spec_s else None},
        "identical_output": identical,
        # A speed number is only meaningful if the outputs match. Never report one
        # for a divergent run.
        "speedup": (round(base_s / spec_s, 3) if spec_s else None) if identical else None,
        "baseline_text": base.text,
        "spec_text": spec_text,
    }


# --------------------------------------------------------------------------- #
# Rigorous wall-clock timing — only run after the correctness gate passes.
# --------------------------------------------------------------------------- #

def _timed_call(fn: Callable[[], Any], validate: Callable[[Any], bool]) -> tuple[float | None, Any]:
    """Time one call; return ``(seconds, result)`` or ``(None, result)`` if invalid.

    A failed/short-circuited call must never be counted as a fast sample, so an
    invalid result yields ``None`` seconds and is dropped by the caller.
    """
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    return (dt if validate(result) else None), result


def _summarise(samples: list[float]) -> dict[str, Any]:
    valid = [s for s in samples if s is not None]
    if not valid:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(valid),
        "median": round(statistics.median(valid), 5),
        "min": round(min(valid), 5),
        "max": round(max(valid), 5),
        "stdev": round(statistics.pstdev(valid), 5) if len(valid) > 1 else 0.0,
    }


def run_timed_bench(upstream: str, api_key: str, model: str, prompt: str,
                    capability: EndpointCapability, max_tokens: int = 128,
                    samples: int = 5, draft_chars: int = 64, burst_tokens: int = 16,
                    warm_text: str | None = None, timeout: int = 600) -> dict[str, Any]:
    """Alternating, validated, regime-separated wall-clock — correctness-gated.

    Requires a *usable* capability (caller must have passed the correctness gate).
    For each of three prefix-cache regimes it alternates baseline/spec order across
    ``samples`` runs, drops any run whose output is not byte-identical to a plain
    call (so a fast error can never masquerade as a fast success), and reports
    median with min/max plus the HTTP request count and whether prefix caching was
    observed.

    Regimes (the engine cache cannot be reset through the public API, so they are
    approximated by prompt construction, and labelled as such):

    * ``cold_prefix`` — a unique nonce is prepended each run, defeating prefix reuse;
    * ``warm_prefix`` — the shared prompt is pre-warmed once, then reused;
    * ``fully_cached`` — the exact prompt is warmed immediately before each timed run.
    """
    if not capability.usable:
        return {"error": f"timed bench needs a usable capability, got {capability.status}"}

    def _plain(p: str):
        return generate_burst(upstream, api_key, model, p, max_tokens, timeout=timeout)

    def _spec(p: str):
        mem = LookupMemory()
        if warm_text:
            mem.observe(warm_text)
        return spec_generate(upstream, api_key, model, p, max_tokens=max_tokens, memory=mem,
                             capability=capability, draft_chars=draft_chars,
                             burst_tokens=burst_tokens, timeout=timeout)

    regimes: dict[str, dict[str, Any]] = {}
    for regime in ("cold_prefix", "warm_prefix", "fully_cached"):
        base_times: list[float | None] = []
        spec_times: list[float | None] = []
        spec_requests: list[int] = []
        all_identical = True
        for i in range(samples):
            if regime == "cold_prefix":
                p = f"[{uuid.uuid4().hex}] {prompt}"
            else:
                p = prompt
                _plain(p)  # warm the shared prefix (and, for fully_cached, the exact text)

            # Establish this run's ground truth once, validated.
            truth = _plain(p)
            truth_ok = (not truth.error) and bool(truth.text)

            def _bt(p=p):
                return _plain(p)

            def _st(p=p):
                return _spec(p)

            # Alternate which lane runs first to cancel first-call cache effects.
            # ``t=truth`` binds this iteration's ground truth into the validators.
            def _base_ok(r, t=truth):
                return (not r.error) and r.text == t.text

            def _spec_ok(r, t=truth):
                return r[0] == t.text

            if i % 2 == 0:
                bs, br = _timed_call(_bt, _base_ok)
                ss, sr = _timed_call(_st, _spec_ok)
            else:
                ss, sr = _timed_call(_st, _spec_ok)
                bs, br = _timed_call(_bt, _base_ok)

            if not truth_ok:
                continue
            base_times.append(bs)
            spec_times.append(ss)
            if isinstance(sr, tuple):
                _, st = sr
                spec_requests.append(st.requests)
                all_identical = all_identical and (ss is not None)

        base_sum = _summarise(base_times)
        spec_sum = _summarise(spec_times)
        speedup = None
        if base_sum["median"] and spec_sum["median"]:
            speedup = round(base_sum["median"] / spec_sum["median"], 3)
        regimes[regime] = {
            "baseline_seconds": base_sum,
            "spec_seconds": spec_sum,
            "spec_http_requests_median": (round(statistics.median(spec_requests), 1)
                                          if spec_requests else None),
            "all_outputs_identical": all_identical,
            "speedup_median": speedup if all_identical else None,
        }

    # Prefix caching is "observed" if warming a prefix makes plain decode faster.
    cold = regimes["cold_prefix"]["baseline_seconds"]["median"]
    warm = regimes["warm_prefix"]["baseline_seconds"]["median"]
    prefix_cache_observed = bool(cold and warm and warm < 0.9 * cold)

    return {
        "capability": capability.status,
        "shift": capability.shift,
        "max_tokens": max_tokens,
        "samples": samples,
        "regimes": regimes,
        "prefix_cache_observed": prefix_cache_observed,
        "note": "wall-clock is host-dependent; request count is the portable "
                "cost signal. Regimes approximate cache state via prompt "
                "construction (the engine cache is not resettable via the API).",
    }


def save_bench_results(out_dir: str | Path, *, environment: dict, capability: dict,
                       correctness: dict, benchmark: dict | None = None,
                       raw_texts: dict[str, str] | None = None) -> dict[str, str]:
    """Persist machine-readable results (and raw texts) under ``out_dir``.

    Layout mirrors ``results/spec_phase2/``. Absolute paths and secrets must not
    be placed in ``environment`` by the caller; this only serialises what it is
    given. Returns the map of written files.
    """
    out = Path(out_dir)
    (out / "raw_responses").mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _dump(name: str, payload: Any) -> None:
        path = out / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        written[name] = str(path)

    _dump("environment.json", environment)
    _dump("capability.json", capability)
    _dump("correctness.json", correctness)
    if benchmark is not None:
        _dump("benchmark.json", benchmark)
    for key, text in (raw_texts or {}).items():
        path = out / "raw_responses" / f"{key}.txt"
        path.write_text(text)
        written[f"raw_responses/{key}.txt"] = str(path)
    return written


def run_cost_probe(upstream: str, api_key: str, model: str, prompt: str,
                   ks: tuple[int, ...] = (1, 4, 8, 16, 32), repeats: int = 5,
                   timeout: int = 600) -> dict[str, Any]:
    """Engine physics: scoring k tokens (one round-trip) vs decoding k tokens.

    Warms the engine's prefix cache on ``prompt`` first, then for each k reports
    the **median** (with min/max) decode-k and score-k time over ``repeats``
    validated samples — a single fast/slow outlier no longer sets the number.

    Breakeven acceptance for draft width k ≈ score_k / decode_1: if a verify round
    emits at least that many tokens on average, speculation wins.
    """
    warm = generate_burst(upstream, api_key, model, prompt, 1, timeout=timeout)
    if warm.error:
        return {"error": f"warmup failed: {warm.error}"}

    rows = []
    for k in ks:
        gen = generate_burst(upstream, api_key, model, prompt, k, timeout=timeout)
        if gen.error or not gen.text:
            return {"error": f"decode probe failed at k={k}: {gen.error or 'empty'}"}
        scored_prompt = prompt + gen.text
        decode = _sample(repeats, lambda k=k: generate_burst(
            upstream, api_key, model, prompt, k, timeout=timeout), lambda r: not r.error)
        score = _sample(repeats, lambda sp=scored_prompt: score_completion(
            upstream, api_key, model, sp, timeout=timeout), lambda r: not r.error)
        rows.append({
            "k": k,
            "decode_s": decode,
            "score_s": score,
            "score_over_decode": (round(score["median"] / decode["median"], 3)
                                  if decode["median"] else None),
        })
    decode_1 = rows[0]["decode_s"]["median"] if rows and rows[0]["k"] == 1 else None
    for row in rows:
        row["breakeven_accept"] = (
            round(row["score_s"]["median"] / decode_1, 2) if decode_1 else None
        )
    return {"rows": rows, "decode_1_s": decode_1}


def _sample(repeats: int, fn: Callable[[], Any], validate: Callable[[Any], bool]) -> dict[str, Any]:
    times: list[float | None] = []
    for _ in range(max(1, repeats)):
        dt, _ = _timed_call(fn, validate)
        times.append(dt)
    return _summarise(times)


def format_bench(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"error: {result['error']}"
    b = result["baseline"]
    if not result.get("spec_available", True):
        return "\n".join([
            f"capability: {result.get('capability')} — {result.get('capability_detail', '')}",
            f"baseline : {b['tokens']} tok in {b['seconds']}s  ({b['tok_s']} tok/s, 1 request)",
            result.get("note", "speculation unavailable on this endpoint."),
        ])
    s = result["spec"]
    lines = [
        f"capability: {result.get('capability')} (shift={result.get('shift')})",
        f"correctness gate: {'PASS (byte-identical)' if result['identical_output'] else 'FAIL — divergent'}",
        f"baseline : {b['tokens']} tok in {b['seconds']}s  ({b['tok_s']} tok/s, 1 request)",
        f"spec     : {s['tokens_total']} tok in {s['seconds']}s  ({s['tok_s']} tok/s, "
        f"{s['requests']} requests, {s['tokens_per_request']} tok/request)",
        f"           draft_accepted/verify={s['draft_tokens_accepted_per_verify']}  "
        f"emitted/verify={s['tokens_emitted_per_verify']}  "
        f"draft={s['tokens_accepted']} corr={s['tokens_correction']} "
        f"bonus={s['tokens_bonus']} burst={s['tokens_burst']} "
        f"seam_fallbacks={s['seam_fallbacks']} zero_accept_rounds={s['verify_rounds_zero_accept']}",
    ]
    if result["identical_output"]:
        lines.append(f"speedup (round-trip ratio, host-dependent): {result['speedup']}x")
    else:
        lines.append("speedup: WITHHELD — output diverged, no speed number is valid.")
    return "\n".join(lines)


def format_cost_probe(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"error: {result['error']}"
    lines = ["  k | decode k med (s) | score k med (s) | score/decode | breakeven accept",
             "----|------------------|-----------------|--------------|-----------------"]
    for r in result["rows"]:
        lines.append(f"{r['k']:>3} | {r['decode_s']['median']:>16} | {r['score_s']['median']:>15} | "
                     f"{r['score_over_decode']:>12} | {r['breakeven_accept']}")
    lines.append("(medians over validated samples; breakeven accept = tokens a verify "
                 "round must emit to beat sequential decode; lower is better)")
    return "\n".join(lines)
