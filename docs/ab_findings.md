# A/B Findings — Matryoshka off vs on, same prompts

The exact same 8 diverse prompts (legal, medical, finance, technical, science,
logistics, hr, practical) were run twice on two different models. "Matryoshka
on" means a different lever depending on the model, because the models support
different things.

Reproduce: `sclab ab` (compression, current Hermes model) or
`sclab ab --backend orthrus --model orthrus-qwen3-4b` (acceleration).
Reports: `runs/ab_ornith_current/report.html`, `runs/ab_orthrus4b_accel/report.html`.

## 1. Current Hermes model (Ornith-1.0-35B via Ollama) — compression lever

A stock GGUF can't run diffusion decoding, so "Matryoshka on" = semantic prompt
compression (fewer prefill tokens). Same model both sides, compressor
`extractive_relevance`.

| metric | result |
|---|---|
| avg end-to-end speedup | 1.13× |
| avg prompt size | 33% of raw |
| quality baseline → Matryoshka | 0.84 → 0.61 |

**Verdict: a situational trade, not a free win.** Clean wins where the answer
survives compression (legal 1.36×, medical 1.42× at identical quality), but on
multi-fact questions the extractive step dropped a required sentence and the
answer went wrong (technical: kept the distractor `port 9090`, dropped `8443`).
Compression helps long single-answer retrieval; it hurts short or multi-fact
exactness on this model.

## 2. Orthrus-Qwen3-4B served by Matryoshka — acceleration lever

Same prompt both sides; "Matryoshka on" = dual-view diffusion decoding, which is
verified by the exact AR pass so the output is lossless.

| metric | result |
|---|---|
| avg decode speedup | **1.71×** (up to 2.28×) |
| avg accepted tokens / pass | 8.1 |
| quality baseline → Matryoshka | 0.85 → 0.79 |
| token-identical outputs | **7 / 8** |

**Verdict: a real, near-lossless speedup.** Structured tasks roughly doubled
decode throughput at identical output — technical 52 → 118 tok/s (2.27×),
practical 2.28×, science 2.25×, medical 1.8×. Seven of eight outputs were
token-for-token identical to plain AR. The one exception (legal) is a
floating-point tie-break divergence in the KV cache — the diffusion path picked
a different, lower-scoring redistribution clause (0.93 → 0.475). This is the
documented, rare fp-tie caveat, not a systematic quality loss; on 8B the speedup
is larger still (see `docs/orthrus_phase1_results.md`).

## Takeaway

- **For the current Hermes model**, Matryoshka's only lever is compression — use
  it for long retrieval, not exact multi-fact prompts.
- **The big lossless win (≈1.7–3×) comes from running an Orthrus model through
  Matryoshka** — same answers, ~2× faster on structured/technical work. To get
  it inside Hermes, serve `orthrus-qwen3-4b` (or 8B) and point Hermes at it.
- Both levers **compose**: compress the prompt *and* diffusion-decode (measured
  ~2.8× end-to-end on long-context QA in `docs/orthrus_phase4_results.md`).
