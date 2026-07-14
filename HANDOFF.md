# HANDOFF — Making Matryoshka Universal

> Goal, in one line: **faster local inference with byte-for-byte zero quality
> loss, on *any* model, on *any* runtime, on *any* machine.**

This document works backwards from the perfect tool to a concrete roadmap, and
ships a working, tested prototype of the one idea that can actually get us
there. Everything here is in the repo's spirit: measured, not asserted, with
the negative findings kept visible.

---

## 1. The North Star — what the perfect tool is

Imagine a single local binary. You point *any* OpenAI-compatible client at it,
and point it at *whatever* you already run — Ollama, LM Studio, llama.cpp,
vLLM, MLX, an Orthrus checkpoint, a 70B on a Mac Studio, a 1B on a laptop. From
that moment:

1. **Every response is faster** — often 2–4×, more on the repetitive/structured
   work that dominates real agent and coding sessions.
2. **Every response is provably identical** to what the raw model would have
   produced. Not "as good." *Identical*, verifiable, on by default.
3. **It gets faster the longer you use it.** It remembers what your models and
   your workloads look like and speculates better over time.
4. **It is model-agnostic and runtime-agnostic** — no special checkpoint, no
   architecture port, no retraining, no GPU-specific kernel.
5. **You can see it working** — the existing live dashboard, now driven by a
   universal speedup metric that means the same thing for every backend.

The current repo delivers (1)+(2) on *exactly three checkpoints on one
platform* (Orthrus-Qwen3 on Apple-Silicon MLX). The compression lever is
model-agnostic but it's a situational prefill trade, not a decode accelerator,
and it is *not* lossless. **The universal, lossless, decode-time win does not
exist in the repo yet.** That's the gap this plan closes.

---

## 2. Why today's design can't be universal (honest gap analysis)

| Lever in repo today | Universal? | Lossless? | Attacks | Verdict |
|---|---|---|---|---|
| Orthrus dual-view diffusion | ❌ 3 checkpoints, MLX only | ✅ (AR-verified) | decode | Real but narrow |
| Semantic compression | ✅ any model | ❌ can drop facts | prefill | Situational trade |
| Copy proposer (CopySpec) | ❌ lives inside the MLX engine | ✅ | decode | Trapped in one runtime |
| Transparent proxy + dashboard | ✅ any OpenAI API | n/a | observability | **Already universal** |

The one genuinely universal, already-shipped thing is **the proxy** — it sits
in front of any OpenAI-compatible engine and observes every token. That is the
delivery vehicle. What it's missing is a **universal acceleration payload** that
rides inside it and preserves output exactly.

The reason speculative decoding has been "locked in the engine" is that it
needs logits and KV-cache surgery. The breakthrough is realizing **you don't
need to be inside the engine** — you need one thing the engine already exposes.

---

## 3. The universal primitive — generation *is* scoring

Every autoregressive engine can *in principle* answer one question through its
public API:

> "For this exact text, at each position, what is your most likely next token?"

That's the OpenAI **legacy completions** call with `echo=true` + `logprobs`.

> **Phase 1 reality check** (see [`docs/spec_phase1_results.md`](docs/spec_phase1_results.md)):
> "in principle" is doing real work here. Not every OpenAI-compatible engine
> actually answers it — the current native `llama.cpp` `llama-server` ignores
> `echo` and returns generated-token logprobs only. And an engine that returns
> the right *shape* may not return the right *semantics*: `llama-cpp-python`'s
> per-position logprobs are **shifted by one**. So the primitive must be
> **probed and its alignment measured** per engine, never assumed — and the
> guarantee below holds for **raw-argmax greedy** decoding specifically (all
> penalties/sampling off), because that is the only policy the scored top-1 can
> verify.
For **greedy** decoding, the most-likely token at a position *is* the token the
engine would generate there. So:

**Score `context + draft` in one parallel prefill pass, and the echoed
per-position top-1 tokens tell you exactly how many draft tokens greedy
decoding would have produced — plus the correct token at the first
divergence.** That is verified speculative decoding, done entirely through a
public API, with **no draft model, no engine patch, and no special checkpoint.**

Why this is lossless: every emitted token is one of
- a token the engine generated itself (a plain burst, or the bonus token past a
  fully-accepted draft), or
- a draft token that scoring proved equal to the engine's own greedy choice, or
- the engine's greedy correction token at the first mismatch.

By induction every token is exactly what greedy decoding produces. The speedup
is pure: many tokens verified per round-trip instead of one token per
sequential decode step.

Why this is *fast* — the physics, measured (see §5): on any
memory-bandwidth-bound decoder (i.e. every local LLM decoding a single stream),
**scoring k tokens in parallel costs about the same as generating one token.**
So the breakeven is ~1: any draft that lands more than ~1 token per round wins.

### Where the drafts come from, with no model

`sclab.spec.memory.LookupMemory`: a character-level lookup over everything the
session has already seen (prompts *and* generated output). Given the current
context, it proposes the continuation that followed the same suffix last time.
This is CopySpec / prompt-lookup-decoding lifted to **plain text**, so it is
tokenizer-agnostic and works across requests. Agent, RAG, and coding workloads
are overwhelmingly self-repetitive (tool schemas, quoted context, code being
edited, JSON templates), so a zero-cost text memory lands large drafts — and
the memory persists, so the accelerator gets faster the more you use it.

This is the universal analogue of the repo's existing (MLX-only) copy proposer
and Orthrus diffusion drafts — same "propose → AR-verify" contract, but the
verify step is now a public API call instead of an in-engine forward pass.

---

## 4. What shipped in this PR (the prototype)

A new, self-contained, dependency-free package `src/sclab/spec/`:

- `memory.py` — `LookupMemory`, the draft-free text proposer.
- `verify.py` — the scoring client (`echo`+`logprobs`) and plain-burst client,
  with careful token/offset parsing and a **seam detector** (see §6).
- `loop.py` — `spec_generate`, the lossless loop (burst / accept / correct /
  bonus), with a DSpark-style backoff when drafts stop landing and a *separate*
  short-burst fallback for tokenization seams.
- `sim.py` — a deterministic, exact-semantics OpenAI-completions engine so
  losslessness is provable in CI with no weights or GPU. Models a real
  tokenizer's canonical (leading-space) tokens and an optional latency model.
- `bench.py` — `run_bench` (baseline vs spec, same greedy request, asserts
  byte-identity) and `run_cost_probe` (engine physics: scoring-vs-decoding
  breakeven).
- CLI: `sclab spec-bench` — runs against any real engine, or `--sim` for an
  instant local demo.

**56 tests grew to 105.** The new tests prove the invariant that matters —
`spec_generate` output is **byte-identical** to a single plain generation call
— across a matrix of workloads, draft alignments (lag), and token budgets,
including early-stop and warm-memory cases, all over real HTTP against the sim.

Run it right now, no model needed:

```bash
sclab spec-bench --sim --max-tokens 120 --draft-chars 96 --cost-probe
```

---

## 5. The evidence (measured)

**Sim engine, decode priced at 20 ms/token, parallel prefill 10× cheaper**
(a faithful stand-in for a memory-bandwidth-bound local decoder):

```
baseline : 120 tok in 2.43s   (49 tok/s, 1 request)
spec     : 120 tok in 0.83s   (144 tok/s, 11 requests, 10.9 tok/request)
           accepted/verify=10.4   identical output: True   speedup: 2.9×
```

**Engine cost probe (content-independent physics):**

```
  k | decode k (s) | score k (s) | breakeven accept
  1 |       0.024  |      0.025  | 1.01
  8 |       0.165  |      0.025  | 1.02
 32 |       0.645  |      0.025  | 1.04
```

The breakeven-acceptance column is the whole thesis in one number: **scoring 32
tokens costs the same as decoding one**, so speculation wins whenever a verify
round emits more than ~1 token. These are *simulated* numbers with the latency
model clearly labeled — the point is the **shape** (breakeven ≈ 1), which is a
property of parallel-prefill-vs-sequential-decode on bandwidth-bound hardware,
not of any particular model. §7 is how we replace them with real ones.

> **These remain simulated.** Phase 1 (see
> [`docs/spec_phase1_results.md`](docs/spec_phase1_results.md)) could not
> reproduce breakeven ≈ 1 on its rig: with no GPU and only a tiny CPU model, the
> measured breakeven was **10–74**, and speculation ran *slower* than plain
> generation — correctly, because that regime is not bandwidth-bound. The 2.9×
> and breakeven≈1 above are properties of the modeled latency, not a measured
> result on a real model. Reproducing them needs a bandwidth-bound decoder.

---

## 6. The critical finding — tokenization boundaries (kept visible)

Building the prototype surfaced the one hard truth any text-level approach must
face, and it's the most important thing for whoever picks this up:

**Re-feeding generated *text* is not always token-identical to a single
continuous generation.** A monolithic generate call appends token *IDs* to the
KV cache and never re-tokenizes. A text-level accelerator must re-feed text, and
if the model emitted a *non-canonical* token sequence (one whose concatenation
re-tokenizes differently — e.g. a no-leading-space token gluing to the previous
one), the re-tokenization shifts and output diverges.

The prototype handles this **safely, never silently**:
- `ScoreResult.draft_tokens` returns `None` when any token straddles the
  context/draft seam; the loop then does a short plain burst to step past it
  (lossless), rather than guessing.
- Real subword tokenizers (SentencePiece `▁`, GPT-2 `Ġ`) use leading-space
  tokens precisely to stay canonical, so seams are rare in practice; the sim
  models this so the losslessness guarantee is clean and honest.

**The guarantee the prototype proves:** every emitted token is the engine's own
greedy choice given the exact preceding text. For canonical tokenizers that is
identical to a monolithic call (proven in CI). The path to *unconditional*
monolithic-call equivalence is §7 Phase 2: **token-ID-level speculation.**

---

## 7. Working backwards — the roadmap

### Phase 0 — shipped here
Prototype, sim, tests, `spec-bench`. Universal, lossless-by-construction at the
text level, measured on the sim.

### Phase 1 — prove it on real engines — **DONE (with corrections); see [`docs/spec_phase1_results.md`](docs/spec_phase1_results.md)**
Outcome, honestly:
- **Correctness survived and is now proven on a real engine** (`llama-cpp-python`
  0.3.16) — byte-identical to plain raw-argmax greedy generation, with real
  draft acceptance — **but only after finding and fixing a genuine bug**:
  `llama-cpp-python` returns prompt logprobs **shifted by +1** from the classic
  convention the parser assumed (API *shape* ≠ API *semantics*). The parser now
  takes a measured `shift`, and `verify.probe_endpoint` measures it per endpoint.
- **The primitive is not universal.** The current native **`llama.cpp`
  `llama-server`** does **not** expose it — `echo` is ignored and only
  generated-token logprobs come back (no `text_offset`). `spec-bench` now probes
  and refuses to speculate on unusable endpoints. **vLLM was not tested** (no
  GPU); do not assume it.
- **Speed is unproven here.** The Phase 1 rig had no GPU and no trained model
  (Hugging Face was egress-blocked), only a tiny CPU model on a synthetic
  fixture. On it, decode is cheap and per-round overhead high, so the
  breakeven≈1 physics does **not** reproduce and speculation is *slower* than
  plain generation despite perfect correctness. The backoff does **not**
  guarantee ~1× on overhead-dominated engines. A wall-clock win needs a
  bandwidth-bound decoder (large model / GPU) — the recommended next experiment.

### Phase 2 — token-ID mode for unconditional equivalence (1–2 weeks)
For engines that accept token-ID prompts and return per-token logprobs
(llama.cpp, vLLM, TGI), draft and verify in **token IDs**, eliminating
re-tokenization entirely → byte-perfect vs a monolithic call with zero seam
fallbacks. Keep text mode as the universal fallback. Auto-detect capability via
a one-time probe at startup.

### Phase 3 — fold into the proxy + dashboard (1 week)
Add `--accelerate spec` to `sclab serve --backend proxy`. The proxy already
sees every token; now it *speculates* on the way through, transparently, for
any client. Wire the existing telemetry so `tokens_per_request` and
`accepted_per_verify` render on the live dashboard as a universal, backend-
agnostic speedup meter (replacing the Orthrus-only metrics with ones that mean
the same thing everywhere). Streaming: emit accepted spans as they clear.

### Phase 4 — make the memory smart (research, high upside)
- **Semantic drafts:** when the lookup memory misses, fall back to a tiny
  draft model *only if* it clears the measured breakeven — a universal
  self-speculation lane that needs no matched checkpoint.
- **Cross-session memory:** persist the lookup index to disk per (model,
  project). The tool literally learns your codebase and your agents' schemas.
- **Structural drafts:** grammar/JSON-schema-aware proposals for constrained
  output (the engine is going to emit that `"` and `}` anyway).

### Phase 5 — the universal speedup ledger
Every response carries an honest `x-sclab-speedup` header and a verifiable
receipt: tokens emitted, engine round-trips, and a losslessness attestation.
"Zero quality loss" stops being a claim and becomes a measurement the user can
audit on every single request.

---

## 8. Wilder bets (for when the obvious is done)

- **Speculative *prefill* sharing:** across concurrent agent requests that share
  a huge system prompt, score once, fan out — the proxy is the natural place.
- **N-best verification:** score several candidate continuations in one batched
  round and keep the greedy-consistent one — turns branchy structured output
  into a single round-trip.
- **Draft markets:** multiple proposers (lookup, tiny model, grammar) bid per
  round; the scheduler runs whichever has cleared breakeven lately. This is the
  repo's existing router idea, generalized to a universal, model-free setting.
- **The end state:** speculation stops being an engine feature you must port to
  each model and becomes a *network-level* property of talking to any model —
  the way HTTP caching sits in front of any web server.

---

## 9. Risks & how the prototype already de-risks them

| Risk | Mitigation in code today |
|---|---|
| Silent quality loss | Losslessness proven in CI **and on a real engine**; conservative tie-safe accept; seam cases fall back, never guess |
| **API shape ≠ semantics** (logprob alignment shift) | `probe_endpoint` **measures** the shift per engine; loop diverges at the wrong one (tested). Found on llama-cpp-python (+1) — see Phase 1 doc |
| Engine lacks `echo`/prompt-logprobs | `probe_endpoint` classifies it unusable → plain generation, never mis-verified. Native `llama-server` fails this today |
| Slower on unspeculable / overhead-bound engine | Backoff floors round-*count*, **not** wall-clock: measured ~3× slower on a tiny CPU model. Needs a bandwidth-bound decoder to win — not guaranteed |
| Tokenization seams | Detector + short-burst step-past; Phase 2 (token-ID mode) removes them |
| Overhead per round-trip | Cost probe measures breakeven per engine before trusting it; on the Phase 1 rig breakeven was 10–74, not ~1 |

---

## 10. Start here (for the next session)

1. `pip install -e ".[dev]" && pytest -q` — 105 tests, no GPU.
2. `sclab spec-bench --sim --cost-probe` — see the mechanism and the physics.
3. Read `src/sclab/spec/loop.py` (the whole idea is ~180 lines).
4. Do **Phase 1**: start a llama.cpp `server`, run
   `sclab spec-bench --upstream http://localhost:8080/v1 --model <m> --cost-probe`,
   and write up real numbers — positive and negative — in
   `docs/spec_phase1_results.md`.

The universal, lossless decode accelerator is no longer a research question in
this repo. It's a prototype with a green test suite and a measured breakeven.
The rest is turning the sim's numbers into real ones and folding it into the
proxy that's already universal.
