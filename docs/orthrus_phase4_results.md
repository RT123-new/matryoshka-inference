# Orthrus Phase 4 — Compression × Diffusion Compose (the multiplier)

The headline experiment of the checklist: does prompt-layer semantic compression
(which shrinks *prefill*) multiply with Orthrus dual-view diffusion decoding
(which accelerates *decode*)? Run entirely through the real sclab CLI on the
`orthrus-mlx` runtime.

## Command (reproducible)

```bash
# diffusion arm
sclab benchmark --runtime orthrus-mlx --model orthrus-qwen3-4b \
  --dataset data/tasks/synthetic_long.jsonl \
  --compressors raw,semantic_brief,extractive_relevance \
  --max-tasks 4 --max-tokens 160 \
  --runtime-options '{"mode":"diffusion","block_size":16}' \
  --out runs/phase4_4b_diffusion
# AR arm: same, with --runtime-options '{"mode":"ar"}'
```

## The 2×2 (Orthrus-Qwen3-4B, long-context factual QA)

| decode | compressor | quality | prompt tok | ratio | wall s | decode tok/s | acc/pass |
|---|---|---:|---:|---:|---:|---:|---:|
| AR   | raw                  | 0.983 | 840 | 1.00 | 0.82 | 48.0 | 1.00 |
| AR   | semantic_brief       | 0.983 | 323 | 0.38 | 0.48 | 51.2 | 1.00 |
| AR   | extractive_relevance | 0.983 | 200 | 0.24 | 0.45 | 51.4 | 1.00 |
| DIFF | raw                  | 0.983 | 840 | 1.00 | 0.74 | 85.2 | 8.71 |
| DIFF | semantic_brief       | 0.983 | 323 | 0.38 | 0.35 | 108.2 | 10.38 |
| DIFF | extractive_relevance | 0.983 | 200 | 0.24 | **0.29** | **119.4** | 11.00 |

## Takeaway — the gains multiply, quality holds

- **Quality is identical (0.983) in every cell.** Neither compression nor
  diffusion decoding hurt answer quality on these long-context QA tasks.
- **Compression alone** (extractive + AR): 0.82 → 0.45 s = **1.8×**, by cutting
  the prompt to 24% of raw (less prefill).
- **Diffusion alone** (raw + DIFF): 0.82 → 0.74 s, and decode throughput 48 →
  85 tok/s. (On these short, source-grounded answers the drafter accepts 8.7
  tokens/pass — extractive answers are highly predictable from context, unlike
  the free-form prose that regressed in Phase 1.)
- **Both together** (extractive + DIFF): 0.82 → **0.29 s = 2.83× end-to-end**,
  decode 48 → **119 tok/s (2.5×)**, at the same 0.983 quality. The corner of the
  2×2 is where prefill compression and decode acceleration stack.

## Why this is the right architecture

Compression and diffusion attack *different* bottlenecks:

```
raw prompt ──[semantic compressor]──► short prompt ──► Orthrus
              (attacks PREFILL)                         (attacks DECODE
                                                          via draft+verify)
```

They do not compete for the same speedup, so they compose almost
multiplicatively. On a memory-bandwidth-bound Mac this is the most effective
lossless local-inference stack found in this project so far.

## Honest caveats

- Small slice (4 long-context tasks) to prove the wired path within budget;
  scale to the full 30-task set for publication numbers (`--max-tasks` off).
- The win is workload-shaped: diffusion helps here because source-grounded
  answers are highly acceptable. For free-form generation, gate diffusion off
  (see Phase 1b rule of thumb). Compression's win is independent of that and
  applies whenever the input is long.
- These are 4B numbers; 8B should widen the diffusion margin further (AR baseline
  is more bandwidth-bound). Not yet run.
