"""Measure API-level speculation against any OpenAI-compatible engine.

Two measurements, both honest by construction:

* ``run_bench`` — the same greedy request executed twice: once as one plain
  generation call (exactly what a normal client pays) and once through the
  speculation loop. Reports wall-clock, tokens-per-request, and — the part
  that matters — whether the outputs are byte-identical.
* ``run_cost_probe`` — content-independent physics: what does *scoring* k
  tokens cost on this engine (one parallel prefill round-trip, warm cache)
  versus *decoding* k tokens sequentially? The ratio gives the breakeven
  acceptance rate for this engine/host, no lucky workload required.
"""

from __future__ import annotations

import time
from typing import Any

from sclab.spec.loop import spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.verify import EndpointCapability, generate_burst, probe_endpoint, score_completion


def run_bench(upstream: str, api_key: str, model: str, prompt: str,
              max_tokens: int = 256, draft_chars: int = 64, burst_tokens: int = 16,
              warm_text: str | None = None, capability: EndpointCapability | None = None,
              timeout: int = 600) -> dict[str, Any]:
    """Baseline (one plain call) vs the speculation loop, same greedy request.

    Probes the endpoint first (unless a ``capability`` is supplied) and only
    speculates when it is proven usable; against an unusable endpoint it reports
    the baseline and the reason, never a mis-verified speculative result.
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
        max_tokens=max_tokens, memory=memory, shift=capability.shift or 0,
        draft_chars=draft_chars, burst_tokens=burst_tokens, timeout=timeout,
    )
    spec_s = time.perf_counter() - t0

    identical = spec_text == base.text
    return {
        "capability": capability.status,
        "shift": capability.shift,
        "spec_available": True,
        "baseline": baseline,
        "spec": {"seconds": round(spec_s, 4), **stats.summary(),
                 "tok_s": round(stats.tokens_total / spec_s, 2) if spec_s else None},
        "identical_output": identical,
        "speedup": round(base_s / spec_s, 3) if spec_s else None,
        "baseline_text": base.text,
        "spec_text": spec_text,
    }


def run_cost_probe(upstream: str, api_key: str, model: str, prompt: str,
                   ks: tuple[int, ...] = (1, 4, 8, 16, 32), repeats: int = 3,
                   timeout: int = 600) -> dict[str, Any]:
    """Engine physics: scoring k tokens (one round-trip) vs decoding k tokens.

    Warms the engine's prefix cache on ``prompt`` first, then for each k:

    * decode_k: one generation call producing k tokens (k sequential steps),
    * score_k: one echo+logprobs call scoring those same k tokens appended to
      the prompt (parallel prefill of k tokens).

    Breakeven acceptance for draft width k ≈ score_k / decode_1: if a verify
    round emits at least that many tokens on average, speculation wins.
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
        decode_s = _best_of(repeats, lambda k=k: generate_burst(
            upstream, api_key, model, prompt, k, timeout=timeout))
        score_s = _best_of(repeats, lambda sp=scored_prompt: score_completion(
            upstream, api_key, model, sp, timeout=timeout))
        rows.append({
            "k": k,
            "decode_s": round(decode_s, 4),
            "score_s": round(score_s, 4),
            "score_over_decode": round(score_s / decode_s, 3) if decode_s else None,
        })
    decode_1 = rows[0]["decode_s"] if rows and rows[0]["k"] == 1 else None
    for row in rows:
        row["breakeven_accept"] = (
            round(row["score_s"] / decode_1, 2) if decode_1 else None
        )
    return {"rows": rows, "decode_1_s": decode_1}


def _best_of(repeats: int, fn) -> float:
    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


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
        f"baseline : {b['tokens']} tok in {b['seconds']}s  ({b['tok_s']} tok/s, 1 request)",
        f"spec     : {s['tokens_total']} tok in {s['seconds']}s  ({s['tok_s']} tok/s, "
        f"{s['requests']} requests, {s['tokens_per_request']} tok/request)",
        f"           draft_accepted/verify={s['draft_tokens_accepted_per_verify']}  "
        f"emitted/verify={s['tokens_emitted_per_verify']}  "
        f"draft={s['tokens_accepted']} corr={s['tokens_correction']} "
        f"bonus={s['tokens_bonus']} burst={s['tokens_burst']} "
        f"seam_fallbacks={s['seam_fallbacks']} zero_accept_rounds={s['verify_rounds_zero_accept']}",
        f"identical output: {result['identical_output']}",
        f"speedup: {result['speedup']}x",
    ]
    return "\n".join(lines)


def format_cost_probe(result: dict[str, Any]) -> str:
    if result.get("error"):
        return f"error: {result['error']}"
    lines = ["  k | decode k (s) | score k (s) | score/decode | breakeven accept",
             "----|--------------|-------------|--------------|-----------------"]
    for r in result["rows"]:
        lines.append(f"{r['k']:>3} | {r['decode_s']:>12} | {r['score_s']:>11} | "
                     f"{r['score_over_decode']:>12} | {r['breakeven_accept']}")
    lines.append("(breakeven accept = tokens a verify round must emit to beat "
                 "sequential decode; lower is better)")
    return "\n".join(lines)
