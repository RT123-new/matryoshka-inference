# DSpark-Mechanism Integration — Confidence-Scheduled Verification on Apple Silicon

Tests whether DeepSeek's **DSpark** framework (open-sourced 2026; up to 85%
faster serving on V4 models) contributes an additional efficiency boost on top
of this project's compression + diffusion + router stack.

## What DSpark actually does

From the published material: a semi-autoregressive drafter (parallel block +
a rank-256 "Markov head" correction), a **calibrated confidence head** that
predicts each drafted token's survival probability, and a **hardware-aware
scheduler** that admits draft tokens into verification one at a time while
marginal throughput improves — low-confidence guesses simply never enter the
verification queue. Their sharpest reported gain: open-chat acceptance rises
**45.7% → 95.7%** through low-confidence pruning.

Two of its three mechanisms need training (Markov head, confidence head). The
third — **prune what you won't verify, and schedule when to speculate at all** —
is trainable-free here because the Orthrus diffusion pass already yields
per-position logits: the drafter's own softmax probability is a free
(uncalibrated) survival proxy.

## What was implemented (in `orthrus_engine.py`)

1. **`prune_draft` + `prune_tau`** — cut the draft at the point where cumulative
   drafter confidence falls below tau; pruned positions never enter the verify
   pass. Verified still by the exact AR pass → losslessness preserved by
   construction.
2. **`BlockPolicy(mode="scheduled")`** — DSpark-style speculation scheduler:
   when rolling acceptance (measured against the *requested* draft width, so
   pruning can't mask low confidence) collapses, drop to an **AR lane** (plain
   decode, no draft pass) for `backoff_steps` tokens, then re-probe with a small
   block. This also subsumes the planned "online-probe router": it probes
   continuously with real acceptance instead of committing once.
3. Telemetry: `pruned_draft_positions`, `ar_lane_steps` per generation.

## Benchmark integrity note

The first benchmark run was invalidated: a wedged Ollama model (Ornith 35B,
33 GB, 262k context) was stuck "Stopping…" at 100% GPU throughout, and AR
baselines swung 13→39 tok/s. After killing the wedged runner and re-running
with a **paired protocol** (each config bracketed by AR baselines; ratios
computed against local baselines), drift flattened to ±3%. Numbers below are
from the clean paired runs (`runs/dspark_integration/results_paired.jsonl`).

## Result 1 — pruning replicates DSpark's acceptance effect…

| workload | acceptance, fixed16 | acceptance, +prune τ=0.3 |
|---|---:|---:|
| reasoning | 40% | **88%** |
| json | 72% | **88%** |
| prose | 12% | **70%** |
| code | 22% | **77%** |

This is the same effect DSpark reports (45.7→95.7% on open chat): almost all
of what survives pruning gets verified successfully.

## …but wall-clock on single-stream Apple Silicon is a wash

| workload | fixed16 | fixed16+prune τ=0.3 |
|---|---:|---:|
| json | 2.23× | 2.30× |
| reasoning | 1.31× | 1.22× |
| prose | 0.56× | 0.50× |
| code | 0.83× | 0.85× |

**Why:** on the M4 Max the verify pass is memory-bandwidth-bound — its cost
barely depends on width, so narrowing verification from 16 to ~4 positions
saves almost nothing, while forfeiting occasional lucky accepts past the prune
point (reasoning acc/pass 6.70 → 6.03). DSpark's pruning pays off on **batched
GPU serving**, where verification slots are a shared scarce resource and every
pruned slot serves another user's tokens. Single-stream local inference does
not have that economy. This is an architecture-dependent mechanism, not a
universal one — a genuinely useful negative result.

## Result 2 — the scheduler DOES add value: always-on speculation becomes safe

The AR-lane backoff (with acceptance measured pre-prune, and a long backoff)
fixes the one dangerous property of diffusion mode — its 0.5× regression on
free-form prose:

| backoff_steps | prose | json |
|---|---:|---:|
| always-spec (fixed16) | 0.56× | 2.23× |
| scheduled, 24 | 0.76× | 2.49× |
| scheduled, 48 | 0.79× | 2.50× |
| **scheduled, 96** | **0.90×** | **2.50×** |

At backoff 96 the worst case improves from **0.55× to 0.90×** while the winning
workload is completely untouched (backoff never triggers on json; `ar_steps=0`).
On thinking-model output (`enable_thinking=True`, 400 tokens), scheduled mode
is also better than always-spec (0.85× vs 0.80× of AR) — thinking spans behave
like prose, so the scheduler correctly sits them out; the Phase 5.3 hypothesis
("draft thinking harder") is thereby **refuted**: think spans have *low*
drafter acceptance, not high.

Along the way the AR lane was made honest: computing token entropy every AR
step (full-vocab softmax + GPU sync) was taxing the lane meant to be cheap; it
is now computed only on the last backoff step, where the policy consumes it.

## Verdict — does the DSpark mechanism add a boost here?

- **Pruning: no** on single-stream Apple Silicon (wash; keep off by default,
  worth revisiting if this stack ever serves concurrent requests).
- **Confidence scheduling: yes** — `policy="scheduled", backoff_steps=96` turns
  diffusion mode from "2.5× on the right workload, 0.55× on the wrong one" into
  "2.5× on the right workload, 0.90× worst case," making it safe to leave on
  when content is mixed or unknown. The keyword router (1.48× on mixed
  workloads by paying zero probe cost) remains best when the workload is
  predictable per request; the scheduler covers within-request content shifts.

Recommended default: `{"mode":"auto"}` at request level, and
`{"policy":"scheduled","backoff_steps":96}` inside diffusion mode as the
safety net.
