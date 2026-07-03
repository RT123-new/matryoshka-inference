# Final Report — Lossless Local-Inference Acceleration on Apple Silicon

Consolidates the Opus implementation checklist (Orthrus × semantic compression)
into one decision-oriented document. All numbers are from real runs on an
M4 Max / 64 GB; raw data under `runs/`, per-phase detail in the sibling docs.

## The question

Can we make local LLM inference faster **without losing quality**, by combining
two independent levers?

- **Semantic compression** shrinks the *input* the model must prefill.
- **Orthrus dual-view diffusion** proposes multiple tokens per step and verifies
  them with the exact autoregressive (AR) pass, accelerating *decode*.

North-star metric: **accepted tokens per verification pass** (AR = 1.0).

## What was built (all in `semantic-compression-lab/`)

- `runtimes/orthrus_engine.py` — instrumented Orthrus MLX decode loop (original
  `model_mlx.py` untouched). Pluggable proposers (diffusion + copy), adaptive
  block controller, mode router, full per-step telemetry. Sized the ring KV
  cache to prompt+budget (the stock 2048 default silently corrupts long context).
- `runtimes/orthrus_mlx.py` — `orthrus-mlx` runtime implementing the harness's
  `LLMRuntime`; `mode` diffusion|ar|auto, block/adaptive/copy options.
- CLI `--runtime-options` JSON threading; 11 unit tests; full suite **23 passed**.

## Result 1 — the model-size crossover (the key scientific finding)

Diffusion decoding trades extra compute (a draft forward pass) for fewer
sequential steps. It only pays off when the AR baseline is memory-bandwidth-
bound. On Apple Silicon that means **bigger models win more** — confirmed across
three sizes (best lossless block, greedy):

| workload | 1.7B | 4B | 8B | trend |
|---|---:|---:|---:|---|
| AR baseline tok/s | ~108 | ~50 | ~28 | baseline slows → more to win |
| json_repeat speedup | 2.06× | 2.26× | **3.49×** | widens |
| json_repeat acc/pass | 8.4 | 11.8 | **15.5** | scales with size |
| reasoning speedup | 1.38× | 1.59× | ~1.2× | positive |
| code speedup | 0.83× | 0.90× | **1.11×** | crosses >1 at 8B |
| prose speedup | 0.6× | 0.6× | 0.65× | always <1 |

**Takeaway:** on this hardware, diffusion is a win for structured / repetitive /
reasoning output and the win grows with model size. Free-form prose is a
consistent loss at every size. Recommended block size is **16** (not the
checkpoint default 32) — past 16 the extra draft width costs more than it buys.

## Result 2 — strict losslessness, honestly characterized

Greedy diffusion output is token-identical to greedy AR except at **floating-
point near-ties**: at one code-prompt divergence the two candidates had logits
32.000 vs 31.875 (gap 0.125/32), and the diffusion path actually picked the
*true* argmax while the incremental ring-cache AR baseline drifted to the
runner-up. Orthrus is lossless w.r.t. a batch-consistent AR pass, as designed;
divergences vs token-by-token AR occur only where the model is ~indifferent.

## Result 3 — compression × diffusion compose (~2.8×), confirmed at full scale

Full 2×2 through the real CLI on long-context QA (Orthrus-Qwen3-4B), now on the
**complete 30-task set** (180 result rows, zero errors):

| | AR decode | diffusion decode |
|---|---|---|
| raw prompt | 1.52 s (1.0×), q=0.944 | 1.02 s, q=0.944 |
| **extractive-compressed** | 0.90 s (1.7×), q=0.959 | **0.55 s (2.79×), q=0.959** |

Extractive quality is *slightly higher* than raw (0.959 vs 0.944 — less
distractor text to mislead the model). The two levers attack different
bottlenecks (prefill vs decode) so they stack almost multiplicatively.
Runs: `runs/phase4_4b_{ar,diffusion}_full/`.

## Result 4 — the Matryoshka mode router beats both pure strategies

`mode=auto` routes each request (structured/reasoning → diffusion, prose → AR).
On a mixed workload: **auto 10.89 s vs always-AR 16.13 s (1.48×) vs always-
diffusion 14.47 s (1.33×)**. The router wins by taking the fast path per request
and never paying the prose regression.

## Result 5 — real-world test on the user's own model (Ornith-1.0-35B MoE)

Orthrus needs trained checkpoints, so the **compression layer** was tested
against `ornith-moe` (35B Q6_K) via Ollama. On 10 long-context tasks:
`extractive_relevance` = **same quality as raw (0.972), prompt cut to 28%,
1.36× faster**, zero errors. The one trap that bites a reasoning MoE (dumping its
thinking as the answer) is already avoided by the harness default `think=False`.
It also honestly flagged a wrong answer when abstractive compression dropped the
numbers a comparison needed. Detail: `docs/ornith_moe_real_test.md`.

## What didn't work (negative results, kept visible)

- **Outline-conditioned drafting (5.2):** giving the diffusion drafter a plan in
  context lifted acceptance by **+0.1 points** on prose (17.7→17.8%) — no effect.
  The prose bottleneck is token-level lexical entropy, not missing direction. So
  planning does not rescue prose; the router does.
- **Copy proposer on ordinary prompts:** no gain unless output is genuinely
  repetitive. On its target (repeated JSON) it fires for 40% of tokens and adds
  +7% throughput via cheaper (single-pass) proposals.
- **Adaptive block controller pre-tuning:** overshot to block 32; after capping
  at the sweet spot it matches fixed-16. Mechanically correct, but a fixed 16 is
  simpler and just as fast.
- **Layer-skip self-speculation (5.4):** drafting with the model's own first K
  layers gets **0.3–12.8%** acceptance (K=25→75%) — estimated **0.26–0.41×**, a
  net loss. Untrained early-exit logits are a poor draft distribution; this is
  why Orthrus trains its diffusion view. Rules out the "free" stock-model
  speedup. Detail: `docs/orthrus_phase5_4_layerskip.md`.
- **DSpark-style confidence pruning:** replicates DSpark's acceptance effect
  exactly (reasoning 40→88%) but is a wall-clock **wash on single-stream Apple
  Silicon** — the verify pass is bandwidth-bound, so width is nearly free. The
  mechanism pays on batched serving, not here. The DSpark *scheduler* idea does
  pay: see Result 6. Detail: `docs/dspark_integration_results.md`.
- **Thinking-lane throttle (5.3): hypothesis refuted.** Think spans behave like
  prose (low drafter acceptance), not like boilerplate — the scheduler correctly
  sits them out (0.85× vs 0.80× always-spec, both below AR). Route thinking
  models to AR.
- **Lazy context fault-in (5.1):** implemented (compressed-first, escalate to
  raw on a gold-blind trigger) but the trigger fired **0/10** times — the real
  compression failure was a *confidently wrong* numeric answer that no surface
  heuristic can detect. Fault-in needs token-level uncertainty (logprobs),
  which the Ollama API doesn't expose. `runs/faultin_ornith/`.

## Result 6 — DSpark-style speculation scheduler makes diffusion safe

`BlockPolicy(mode="scheduled")`: when measured acceptance collapses, drop to a
plain-AR lane for `backoff_steps` tokens, then re-probe (this also subsumes the
planned online-probe router). At backoff 96 on 4B: **prose worst case improves
0.55× → 0.90×** while json keeps its full **2.50×** (backoff never triggers).
Always-on diffusion is now nearly safe on unknown/mixed content; the keyword
router remains best when content is predictable per request.

## Recommendation (what to actually run)

1. **Always compress long inputs** with `extractive_relevance` — free ~1.3–1.9×
   at no quality loss, works on any model incl. your Ollama models today.
2. **For Orthrus-capable (MLX) models**, enable `mode=auto` with block 16 — adds
   1.3–3.5× on structured/reasoning/repetitive output, avoids the prose loss.
3. **Prefer larger models** for the diffusion win: the 8B crossover is much
   stronger than 1.7B. Combined stack peaks at ~2.8× on long-context QA.

## Open / next

- 5.4 layer-skip — **tested, ruled out**. 5.2 outline-conditioning — **tested,
  no effect**. 5.3 thinking throttle — **tested, hypothesis refuted**.
  5.1 fault-in — **tested; blocked on logprob access**. Online-probe router —
  **subsumed by the scheduled policy**. Full 30-task Phase 4 — **done (2.79×)**.
- Phase 6 Orthrus-Gemma port — **feasibility memo written, verdict: do not
  build**; check for official Gemma MTP drafters instead
  (`docs/phase6_gemma_port_memo.md`).
- 5.5 two-machine brigade — not testable here (no second Mac); the engine's
  proposer abstraction accepts a network proposer as a drop-in when one exists.
- Genuinely open: fault-in with logprob-based triggers on the MLX runtime, and
  revisiting confidence pruning if this stack ever serves concurrent requests.
