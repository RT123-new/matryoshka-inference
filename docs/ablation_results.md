# Ablation Results: Long-Context Compression Cliff

This phase tested prompt-layer compression only. It does not show or claim model-internal compression.

Evidence used:

- Prior tuned run: `runs/gemma4_latest_long_30_tuned/`
- Final semantic ablation: `runs/gemma4_latest_long_ablation_semantic_brief_fixed/`
- Final extractive ablation: `runs/gemma4_latest_long_ablation_extractive_relevance_fixed2/`
- Dataset: `data/tasks/synthetic_long.jsonl`
- Runtime/model: Ollama, `gemma4:latest`

## What Changed

Two obvious prompt-layer compressor fixes were made before the final reruns:

- `semantic_brief` now respects ablation budgets and spends tight budgets on exact relevant excerpts before derived bullets.
- `extractive_relevance` now has small whole-word intent bonuses for obligation, deadline, cost-comparison, and contradiction questions.

## Budget Results

| compressor | budget | avg quality | prompt ratio | prompt reduction | avg latency factor | pass rate |
|---|---:|---:|---:|---:|---:|---:|
| semantic_brief | 0.2 | 0.949 | 0.360 | 64.0% | 1.81x | 80.0% |
| semantic_brief | 0.3 | 0.949 | 0.392 | 60.8% | 1.61x | 93.3% |
| semantic_brief | 0.4 | 0.952 | 0.402 | 59.8% | 1.92x | 93.3% |
| semantic_brief | 0.6 | 0.952 | 0.405 | 59.5% | 1.86x | 93.3% |
| semantic_brief | 0.8 | 0.952 | 0.405 | 59.5% | 1.81x | 93.3% |
| extractive_relevance | 0.2 | 0.917 | 0.232 | 76.8% | 2.16x | 93.3% |
| extractive_relevance | 0.3 | 0.972 | 0.257 | 74.3% | 1.79x | 100.0% |
| extractive_relevance | 0.4 | 0.972 | 0.264 | 73.6% | 1.77x | 100.0% |
| extractive_relevance | 0.6 | 0.972 | 0.264 | 73.6% | 3.65x | 90.0% |
| extractive_relevance | 0.8 | 0.972 | 0.264 | 73.6% | 1.49x | 90.0% |

## Plain-English Answer

The safest compressor is `extractive_relevance`. It preserves exact source wording, has the best quality after the fix, and reached 100% pass rate at budgets `0.3` and `0.4`.

The best overall budget is `extractive_relevance` at `0.3`. It kept quality high, reduced prompt size by about 74%, and passed every long-context task in this run. If using `semantic_brief`, the safest budget is `0.4`; `0.2` is too tight.

Quality breaks first on reasoning-heavy tasks, not simple lookup. `semantic_brief` still struggles with multi-fact calculation (`cheaper_001_long`) and contradiction wording (`contradiction_001_long`) even when the relevant evidence is present. `extractive_relevance` breaks at budget `0.2` on contradiction detection because it can still drop the needed exact paragraph; at `0.6` and `0.8`, failures were mostly latency-pass failures, not answer-quality failures.

Number/date precision and code/config QA were strong after the fixes. Both compressors reached 100% pass rate for code/config QA. `semantic_brief` averaged 0.980 on number/date precision; `extractive_relevance` averaged 0.958.

The speed gains look real for this long-context prompt-layer setup, but they are noisy. The final runs and the prior tuned run both show average latency factors above 1x for compressed prompts. However, local Ollama contention made absolute timings non-monotonic, so treat the speed result as "compressed long prompts are usually faster here," not as a guaranteed per-task latency win.

## What To Test Next

1. Test `extractive_relevance` more densely around budgets `0.24`, `0.28`, `0.32`, and `0.36`.
2. Add a contradiction-specific retrieval guard that requires both sides of a conflict before answering.
3. Add a calculation-aware compressed prompt for cost comparisons so the model computes totals from preserved figures.
4. Repeat the best budgets with an uncontended Ollama runtime and multiple seeds to separate real latency gains from local runtime noise.
