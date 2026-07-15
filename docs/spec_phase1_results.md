# Spec Phase 1 Results — API-Level Verified Speculation on Real Engines

> **Phase 2 update.** The surface-vs-id gap and the seam-fallback path this
> document flags are addressed by **token-ID mode** — see
> [`docs/spec_phase2_results.md`](spec_phase2_results.md). Phase 2 also hardened
> the text-surface probe (byte offsets, incomplete echo, partial coverage,
> ambiguous alignment and missing bonus are now rejected) and made
> `spec_generate` refuse to speculate without a usable capability. Text-surface
> mode as described below is **conditional and experimental**, proving *surface*
> identity only; token-ID mode is the lane for unconditional equivalence.

This records Phase 1 from `HANDOFF.md`: take the Phase 0 prototype of
*API-level verified speculative decoding* — which was proven lossless only
against a deterministic **simulated** engine — and test it against **real local
inference engines**. The goal was to prove or disprove two claims, honestly:

- **(a) Correctness** — speculative output is **byte-identical** to a single
  plain generation call.
- **(b) Speed** — on a memory-bandwidth-bound decoder, scoring *k* tokens costs
  ≈ decoding one, so any draft landing >1 token/round wins (breakeven ≈ 1.0).

**Bottom line up front.** The core primitive **survived, but only conditionally
and only after a real bug was found and fixed.** API *shape* compatibility is
**not** API *semantics* compatibility: the one real engine that exposes the
primitive (`llama-cpp-python`) returns per-position logprobs **shifted by one**
from the convention the parser assumed, which silently corrupted output until
measured and corrected. The current native `llama.cpp` `llama-server` does not
expose the primitive at all. Correctness (a) is now **proven on a real engine**.
Speed (b) is **not** demonstrable in this environment and remains unproven —
the rig has no memory-bandwidth-bound decoder, and on a tiny CPU model
speculation is *much slower* than plain generation despite perfect correctness.

---

## 1. Environment and exact versions

| | |
|---|---|
| Machine | Linux container, **no GPU**, 4 vCPU, ~15 GB RAM, ~30 GB disk |
| Python | 3.11.15 |
| Engine A | native **`llama.cpp` `llama-server`**, built from source, commit `657e011` (2026-07-14), CPU build |
| Engine B | **`llama-cpp-python` 0.3.16** OpenAI server (`python -m llama_cpp.server`, `--logits_all true`) |
| Models | two **synthetic** GGUF fixtures (see §2) — no trained model was reachable |

**A hard environment constraint, reported because it shaped everything:** the
sandbox's egress policy **blocks Hugging Face and Ollama** (and model CDNs);
PyPI and GitHub are allowed. So no pre-trained GGUF could be downloaded. We
built `llama.cpp` from GitHub and **synthesised** loadable models with a **real
GPT-2 BPE tokenizer** (copied from `llama.cpp`'s bundled `ggml-vocab-gpt-2.gguf`)
and random/hand-set weights. This is enough to exercise the real engine, real
tokenizer, real HTTP API, and real greedy determinism — which is exactly what
the capability probe, the alignment diagnostic, and the losslessness gate need —
but it is **not** enough to claim anything about trained-model quality or
realistic acceptance rates. Every number below is labeled accordingly.

## 2. The synthetic fixtures (and their limits)

Both are real `llama` architecture (4 layers, d=128), GPT-2 BPE tokenizer,
served identically by both engines. Builders live in the PR (`scripts/`).

- **`tiny-llama-gpt2`** — random weights. **Pathological and unusable as a
  fixture:** near-tied logits everywhere, greedy collapse onto token id 0
  (`"!"`), and non-canonical repeated punctuation that re-tokenizes differently
  when re-fed (a 16-token tail re-tokenized to 2 tokens). It is kept only as the
  honest illustration of *what breaks a text-level approach*; it is **not** a
  model and proves nothing about real performance.
- **`tiny-cycle-gpt2`** — hand-set weights so the residual carries the current
  token embedding through near-zero attention/FFN and the LM head maps each of
  64 canonical leading-space word tokens to the next. Result: **deterministic,
  confident (huge argmax margin), canonical (re-tokenizes to itself), periodic**
  greedy output. This is a legitimate *integration fixture* — it makes the
  engine behave the way a **trained** model behaves on the axes the primitive
  depends on (confidence + canonical tokens) — but it is **not** a language
  model, and its acceptance rates are a property of its periodicity, **not** of
  any workload. **No speedup or workload claim may rest on it.**

## 3. Capability matrix — behaviour, not field names

| Engine | `echo` honored? | prompt logprobs? | classic `{tokens,token_logprobs,top_logprobs,text_offset}`? | positional alignment | **usable?** |
|---|---|---|---|---|---|
| native `llama-server` (`657e011`) | **No** (silently ignored) | **No** (generated tokens only) | **No** — returns `logprobs.content` | — | **No** |
| `llama-cpp-python` 0.3.16 | Yes | Yes | Yes | **shifted by +1** (§5) | **Yes**, after the shift fix |
| vLLM | not tested (no GPU) | — | uses `prompt_logprobs` (different shape) | — | untested |
| Ollama | not tested (blocked) | `/v1/completions` does not echo prompt logprobs (per upstream) | — | — | expected No |

`spec-bench` now runs a **behavioural probe** (`verify.probe_endpoint`) that
classifies an endpoint and **refuses to speculate** unless it proves it echoes
the prompt, returns prompt-position candidates, has a measurable shift under
which a *known* greedy continuation verifies end-to-end, and exposes the bonus
position. Native `llama-server` fails this cleanly → falls back to plain
generation with a message. We did **not** test vLLM (no GPU) and do not claim it.

## 4. Native `llama-server` — a clean negative result

The reference target from the plan does **not** currently expose the primitive
through its OpenAI-compatible endpoint. Measured on commit `657e011`:

- `POST /v1/completions` with `echo:true` returns HTTP 200 but the prompt is
  **not** echoed; the request routes to `post_completions_oai → handle_completions_impl`,
  which does not honor `echo` (the `oaicompat_completion_params_parse` guard
  that would reject it is dead code in this build).
- `logprobs` come back as `logprobs.content` (chat-style, per **generated**
  token) — there is **no** `tokens`/`token_logprobs`/`top_logprobs`/`text_offset`,
  and `text_offset` appears **nowhere** in the source tree.
- The native `/completion` endpoint's `n_probs` likewise reports probabilities
  for generated tokens only (`populate_token_probs` runs during decode).

So there is no way to get per-position top-1 over the prompt/draft from this
server. This is a genuine limitation of "works with llama.cpp via its public
API," and it is the single most consequential Phase 1 finding for the "universal"
framing. (The lookup-memory + plain-burst path still runs through it, but with
no verification lane there is no speedup — and the probe now says so.)

## 5. The alignment diagnostic — API shape ≠ API semantics

`llama-cpp-python` returns the classic response *shape*, and the existing parser
reads it without error. **That is not the same as the fields meaning what the
loop assumed.** The Phase 0 parser assumed the OpenAI convention where index `i`
describes token `i` (the distribution that *produced* it). Measured against the
`tiny-cycle-gpt2` fixture, over a **known raw-argmax greedy continuation**:

```
argmax(top_logprobs[i]) == echoed token S[i]     :  0 / 30
argmax(top_logprobs[i]) == next token   S[i+1]   : 24 / 29
```

Cross-checked against ground-truth logits from the in-process
`llama_cpp.Llama(logits_all=True)` API, `token_logprobs[i]` equals
`logP(S[i] | logits[i])`, **not** `logP(S[i] | logits[i-1])`. In other words:

> **`llama-cpp-python` pairs echoed token `i` with `logits[i]` — the model's
> distribution *after* consuming token `i`, i.e. the prediction for token
> `i+1`. Everything in the logprobs payload is shifted by +1 versus the classic
> convention.** The prediction that produced token `i` lives at index `i-1`.

This is exactly the "#1 integration risk" the handoff flagged, and it is worse
than a field rename: it is a silent off-by-one that **drops tokens**. With the
old parser, a byte-identical short run would begin to diverge the moment the
verification lane engaged (e.g. baseline `…would off res…` vs spec `…would
res…` — the ` off` token dropped). It was **not** caused by `repeat_penalty`,
re-tokenization, or fp near-ties (all ruled out by isolated tests); it was the
alignment.

**Fix.** `verify._parse_logprobs` takes a `shift`, and the prediction for token
`i` is read from `top_logprobs[i - shift]`. `verify.probe_endpoint` **measures**
the shift per endpoint (0 = classic/OpenAI/the sim, 1 = llama-cpp-python) by
checking which shift makes a known greedy continuation verify. A verbatim
response is checkpointed at `tests/fixtures/llamacpp_python_echo_shift.json` as a
regression guard. The sim can now emulate **both** conventions, and the loop is
lossless **only** at the correct shift and provably **diverges** at the wrong
one (`tests/test_spec_alignment.py`) — the guard that pins the whole fix.

Two more conservatism fixes went in alongside it, because losslessness is the
gate:

- **`is_greedy` now requires exact, unambiguous identity.** A tie at the top of
  the candidate list, a missing candidate list, or an empty/special surface all
  count as *not greedy* → the loop corrects or falls back, never guesses. The
  old behaviour of accepting a differently-spelled token because two float
  logprobs printed equal is gone.
- Because text-level scoring returns **surfaces, not token ids**, verification
  can only establish *surface* identity. Duplicate surfaces hiding different ids
  are indistinguishable here; unconditional token-*id* equivalence needs Phase 2
  (token-ID mode). This is documented, not hidden.

## 6. Correctness gate — PASSED on a real engine

With the probe-measured shift and conservative acceptance, on `llama-cpp-python`
serving `tiny-cycle-gpt2`:

| prompts × budgets | identical to plain generation? | real accepted **draft** tokens/verify |
|---|---|---|
| 5 prompts × {40} | ✅ all | 0 (period 64 > 40 → all bursts) |
| 5 prompts × {200} | ✅ all | 12.4 |
| 5 prompts × {300} | ✅ all | 12.8 |
| 4 prompts × {1,16,64,200} (opt-in test) | ✅ all | up to 18.8 (warm) |

`identical_output` is asserted byte-for-byte against a single
`generate_burst(max_tokens=N)`. Accepted counts are **real draft tokens**, not
corrections — the telemetry was corrected (§7). The wrong shift diverges; the
native server is refused. This is the whole value proposition, now measured on a
real engine rather than the sim. (It rests on the fixture's confident, canonical
output — the property a trained model would provide — **not** on trained-model
behaviour, which we could not obtain.)

## 7. Telemetry corrected

The old `accepted_per_verify` folded accepted drafts **plus** corrections
**plus** bonus tokens into one number, so a run that accepted **zero** drafts
but emitted one correction per round read as "1.0 accepted/verify" — which is
how the pre-fix bug hid. Metrics are now disaggregated:
`draft_tokens_accepted_per_verify` (the real win), `tokens_emitted_per_verify`
(the old broad quantity), `corrections_per_verify`, `bonus_tokens_per_verify`,
and `verify_rounds_zero_accept`. The gate asserts **real draft acceptance > 0**,
not merely one emitted token per round.

## 8. Cost & speed — real engine, **synthetic weights**, and an inverted result

> These are **real-engine integration and cost measurements using synthetic
> weights on a tiny CPU model**. They are **not** evidence of trained-model
> workload performance, and they do **not** reproduce the breakeven≈1 physics —
> for an honest reason.

**Cost probe** (`tiny-cycle-gpt2`, `llama-cpp-python`, 4 vCPU, best-of-5):

```
  k | decode k (s) | score k (s) | score/decode | breakeven accept
----|--------------|-------------|--------------|-----------------
  1 |     0.0046   |    0.0481   |     10.38    |   10.5
  8 |     0.1014   |    0.1214   |      1.20    |   26.4
 16 |     0.1971   |    0.1858   |      0.94    |   40.4
 32 |     0.4213   |    0.3423   |      0.81    |   74.4
```

The Phase 0 sim predicted **breakeven ≈ 1**. The real tiny-CPU number is
**10–74**. The physics thesis is *not wrong* — it is a statement about
**memory-bandwidth-bound** decoders, where one sequential decode step is
expensive. This rig is the opposite regime: a tiny model on CPU decodes a single
stream extremely fast (~570 tok/s), while each scoring round-trip pays a large
**fixed** cost (HTTP + the Python server + a full-prompt `logits_all` pass). So
scoring one token costs ~10× decoding one, and speculation must land *tens* of
tokens per round just to break even.

**End-to-end wall clock** (byte-identical in every row):

| config | identical? | accepted draft/verify | speedup |
|---|---|---|---|
| warm memory, 256 tok (fully speculable) | ✅ | 18.8 | **0.027×** (~37× slower) |
| cold memory, 256 tok | ✅ | 18.7 | 0.03× |
| cold memory, 48 tok (unspeculable within request) | ✅ | 0 | 0.31× (~3× slower) |

Even with **near-perfect acceptance** and **perfect correctness**, speculation
is dramatically slower here — because the baseline generates all N tokens in one
cheap call while the loop pays many high-overhead round-trips. Note too that the
"backoff floors at ~1×" claim from Phase 0 does **not** hold on an
overhead-dominated engine: the unspeculable case still ran ~3× slower, because
chunked bursts are multiple HTTP calls versus the baseline's one. Burst
granularity is a real tuning knob on such engines, not a free floor.

**This is the honest converse of the thesis, and it is the expected result for
this hardware.** A wall-clock win requires a decoder where one decode step is
expensive relative to a parallel prefill — a large model, and/or a GPU, and/or a
server with cheap incremental scoring and prefix caching. None were available.

## 9. What works / what doesn't

**Works (measured on a real engine):**
- The propose→verify **mechanism** and its **losslessness**: byte-identical to
  plain raw-argmax greedy generation, with real draft tokens accepted.
- A **behavioural capability probe** that measures alignment and refuses
  unusable endpoints instead of silently mis-verifying them.
- Conservative, tie-safe acceptance; corrected, honest telemetry.

**Doesn't (or unproven):**
- **Native `llama.cpp` `llama-server`**: primitive unavailable — no usable path.
- **"Any OpenAI-compatible engine"**: false as stated. Two engines, same shape,
  **different semantics**; one unusable. Capability must be probed, not assumed.
- **Speed**: no wall-clock win in this environment; breakeven≈1 unreproduced.
  Unproven, not disproven — it needs a bandwidth-bound decoder.
- **Trained-model behaviour** (realistic acceptance on JSON/code/RAG/prose):
  untested — no trained model was reachable. The pathological random fixture is
  a warning, not a workload.
- **Unconditional monolithic-call equivalence**: text mode gives *surface*
  identity for raw-argmax greedy only; token-ID mode (Phase 2) is required for
  token-exact equivalence and would also remove the seam-fallback path.

## 10. Remaining blockers & the recommended next experiment

1. **Get a small trained GGUF to a usable engine** (policy-compliant model
   access, or a machine that can reach Hugging Face). Repeat §6 on e.g.
   Qwen2.5-0.5B/1.5B-Instruct to measure *realistic* acceptance on JSON, code,
   RAG-with-quoting, and prose — and confirm prose stays lossless (it will) and
   near ~1× only if per-round overhead is low.
2. **Run on a bandwidth-bound decoder** (a larger model and/or a GPU, ideally
   vLLM with automatic prefix caching, whose `prompt_logprobs` shape and
   alignment must be probed and almost certainly need their own adapter) to test
   the breakeven≈1 speed thesis where it can actually hold.
3. **Phase 2 token-ID mode** for engines that accept token-ID prompts and return
   per-token logprobs, to reach byte-perfect monolithic-call equivalence with no
   seam fallbacks — and to sidestep the surface-vs-id ambiguity §5 documents.

## Gate status

- ✅ **Correctness gate met on a real engine** (`llama-cpp-python`), byte-identical
  with real draft acceptance, after finding and fixing a genuine alignment bug.
- ✅ Unsupported engines (native `llama-server`) fail the probe cleanly and fall
  back to plain generation.
- ⛔ **Speed gate not met** and not attributable — no bandwidth-bound decoder was
  available; measured wall-clock is a large *slowdown* on the tiny CPU model.
- 🔒 The original "any OpenAI-compatible engine, universal" claim is **narrowed**
  to "engines that pass a behavioural capability + alignment probe," which today
  means `llama-cpp-python`, not native `llama-server`.

The primitive is real and now lossless against a real engine — a stronger claim
than Phase 0's sim-only proof — but "universal" and "faster" are both
**unproven** beyond this rig, and one is contradicted on it. A rigorous negative
is more useful than a preserved thesis.
