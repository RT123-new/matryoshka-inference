# Orthrus Phase 5 — Router, Adaptive Tuning, Copy Proposer (4B)

Follow-ups that close the gaps flagged in Phase 1b and build the most practical
"outlandish" experiment from the checklist (Phase 5.6, the Matryoshka mode
router). All on `chiennv/Orthrus-Qwen3-4B`, M4 Max, greedy, decode tok/s
excludes prefill. Raw data: `runs/orthrus_phase5/results.jsonl`.

## (1) Adaptive controller — gap closed

Phase 1b found the adaptive policy overshot to block 32 (past the throughput
sweet spot of ~16). Fix: grow toward a `structured_block=16` cap instead of
`max_block`. Result — adaptive now tracks fixed-16 on the workloads where
diffusion wins:

| workload | AR tok/s | fixed-16 | adaptive (tuned) |
|---|---:|---:|---:|
| json | 52.8 | 132.5 (2.51×) | 132.6 (**2.51×**) |
| reasoning | 52.6 | 78.3 (1.49×) | 78.4 (**1.49×**) |
| code | 52.9 | 50.4 (0.95×) | 46.1 (0.87×) |
| prose | 52.5 | 33.0 (0.63×) | 29.1 (0.55×) |

Adaptive now matches fixed-16 exactly on json/reasoning. On code/prose it is
slightly behind because it still spends a few steps exploring block sizes — but
those are the workloads you would not run diffusion on at all (that is the
router's job, below).

## (2) Matryoshka mode router — beats always-on AND always-off

`route_mode(prompt)` classifies each request from its prompt shape:
structured/reasoning cues → diffusion; free-form prose → AR (default AR on ties).
Exposed as `--runtime-options '{"mode":"auto"}'`. On a mixed 5-prompt workload
(total wall seconds, prefill included):

| prompt | routed to | AR s | DIFF s |
|---|---|---:|---:|
| json | diffusion | 3.47 | **2.02** |
| code | diffusion | 3.51 | **1.51** |
| reasoning | diffusion | 2.86 | **1.08** |
| prose1 | ar | **2.77** | 4.60 |
| prose2 | ar | **3.52** | 5.27 |
| **TOTAL** | | 16.13 | 14.47 |

**AUTO total = 10.89 s.** That is **1.48× faster than always-AR** and **1.33×
faster than always-diffusion.** The router wins because it takes the fast column
in every row — capturing the 1.7–2.6× diffusion wins on structured/reasoning
prompts while dodging the 1.5–1.7× *regression* diffusion causes on free-form
prose. This is the core payoff: diffusion decoding is a workload-targeted tool,
and a cheap per-request router monetizes it without the downside.

## (3) Copy proposer — fires on repetitive output, cheaper proposals win

On a forced-repetition workload ("repeat this JSON record 12×, incrementing
id"), the CopySpec-style proposer now actually fires:

| config | tok/s | acc/pass | token sources |
|---|---:|---:|---|
| diffusion-only | 186.5 | 16.07 | 240 diffusion |
| diffusion + copy | **199.0** | 12.68 | 144 diffusion, **96 copy** |

40% of tokens (96/240) were served by the copy proposer, and throughput rose
+7% — *even though acc/pass dropped*. The reason is the mechanism: a copy
proposal skips the diffusion draft forward pass entirely (one verify pass
instead of draft+verify), so each copied block is cheaper even when it accepts
fewer tokens. This confirms the Phase 3 design on its intended target
(long structured/repeated output — JSON, tables, file edits).

## Net picture after Phase 5

The strongest lossless local-inference stack this project has produced:

```
prompt ─[compressor: shrinks prefill]─► [router: diffusion vs AR per request]
                                         └► diffusion ─[adaptive block≈16 + copy proposer]─► verify
```

- Compression × diffusion compose to ~2.8× on long-context QA (Phase 4).
- The router adds another 1.3–1.5× by never running diffusion on the wrong
  workload.
- Adaptive block-16 and the copy proposer are the within-diffusion tunings.

## Still open (honest)

- Router uses prompt-keyword classification; an online-probe variant (measure
  acceptance on the first ~16 tokens, then commit) would be more robust for
  ambiguous prompts. Designed, not yet built.
- 8B model not yet run (expected to widen diffusion's margin further).
- Phase 5.1–5.5 (lazy context fault-in, outline-conditioned drafting,
  thinking-lane throttle, layer-skip self-speculation for Gemma) remain as
  designed in the checklist. Layer-skip (5.4) is the path to accelerating the
  user's actual Gemma without an Orthrus port.
