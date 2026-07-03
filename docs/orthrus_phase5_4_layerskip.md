# Phase 5.4 — Layer-Skip Self-Speculation (negative result)

Tests the cheapest hypothetical path to accelerating a **stock** model's decode
without any trained draft head or second model: draft with the model's own first
K layers + final norm + lm_head, verify with the full model (Draft & Verify /
self-speculative decoding). One model in memory, no training.

**Core question:** are untrained early-exit logits accepted often enough by the
full model to yield a speedup?

## Method

On Orthrus-Qwen3-4B (36 layers, used here purely as a generic dense transformer),
draft D=6 tokens greedily from the first K layers, verify with the full model,
accept the longest matching prefix, repeat. Measured draft acceptance rate at
K = 25% / 50% / 75% of layers across four workloads.

## Result — early exit does not draft usefully

| workload | K=25% | K=50% | K=75% |
|---|---:|---:|---:|
| code | 0.0% | 0.0% | 6.9% |
| json | 0.0% | 0.0% | 15.3% |
| reasoning | 0.0% | 1.4% | 20.8% |
| prose | 1.4% | 1.4% | 8.3% |
| **mean** | **0.3%** | **0.7%** | **12.8%** |

Estimated cache-optimized speedup `(accepted+1) / (1 + D·K/N)`:

| K | acc/pass | draft cost | speedup |
|---|---:|---:|---:|
| 25% | 1.02 | 1.5 passes | **0.41×** |
| 50% | 1.04 | 3.0 passes | **0.26×** |
| 75% | 1.77 | 4.5 passes | **0.32×** |

Every configuration is a **net loss** (<1×): the cost of running K layers to draft
far exceeds the value of the handful of tokens the full model accepts.

## Interpretation

Bolting layer-skip drafting onto an already-trained model does not work. This
matches the published finding (MTP-for-inference, EAGLE, Medusa): a pretrained
model's hidden layers are strongly specialised for next-token prediction, so
intermediate-layer logits are a poor draft distribution without a **trained**
early-exit / draft head. This is precisely why Orthrus trains ~16% of parameters
for its diffusion view rather than getting drafting for free.

## What this means for accelerating a stock Gemma / GGUF

Ranked by effort:

1. **n-gram / copy self-speculation** — no training, no second model, works
   today for repetitive output. Already implemented here as `CopyProposer`
   (Phase 3): fires on 40% of tokens on repeated JSON, +7% throughput.
2. **Separate small draft model** — standard speculative decoding; needs a
   compatible small model in memory (extra RAM), no training.
3. **Trained draft head (EAGLE / Medusa) or Orthrus** — best acceptance, but
   requires training the added parameters. This is the route with the real
   lossless multi-token wins measured in Phases 1–5.

**Naive layer-skip is ruled out.** For stock models, use copy self-speculation
for repetitive workloads and semantic compression for long inputs (both work
now); reserve the trained-head routes for when a training run is on the table.
