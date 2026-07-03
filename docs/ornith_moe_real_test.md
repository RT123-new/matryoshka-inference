# Real Test — Semantic Compression with Ornith-1.0-35B MoE via Ollama

A real end-to-end run of the compression system against a genuine local model:
`ornith-moe:latest` (deepreinforce-ai Ornith-1.0-35B, Q6_K, ~28 GB MoE) served
by Ollama on the M4 Max.

## Scope note (important)

The Orthrus dual-view diffusion decoder (Phases 1–5) needs Orthrus's *trained*
MLX checkpoints — it cannot run on a stock GGUF through Ollama. So what is tested
here is the **semantic-compression layer** (the lab's original purpose): compress
the prompt, generate with a real local model, measure quality + latency vs the
raw-context baseline. That layer is model-agnostic and runs against any Ollama
model.

## Behavior probe (bug check first)

A 35B *reasoning* MoE is exactly the kind of model that burns its token budget on
hidden thinking and returns no visible answer (the documented `gemma4` trap).
Verified both paths:

- `think=False` (harness default): clean answer — *"The Aurora dam was completed
  in 1974."*, ~90 tok/s.
- `think=True`: model emits its raw thinking process as the answer and hits the
  length cap with no final answer.

The harness already defaults `think=False`, so it sidesteps the trap. **No new
bugs found; no fix required in the Ollama path.**

## Long-context benchmark (10 tasks, `synthetic_long`, max_tokens=200)

```bash
sclab benchmark --runtime ollama --model ornith-moe:latest \
  --dataset data/tasks/synthetic_long.jsonl \
  --compressors raw,semantic_brief,extractive_relevance \
  --max-tasks 10 --max-tokens 200 --out runs/ornith_long_10
```

| compressor | avg quality | avg prompt % | avg speedup | pass % | decode tok/s |
|---|---:|---:|---:|---:|---:|
| raw | 0.972 | 100.0% | 1.00× | 0% | 88.4 |
| semantic_brief | 0.955 | 37.4% | 1.20× | 60% | 88.7 |
| extractive_relevance | **0.972** | **28.3%** | **1.36×** | 80% | 88.9 |

- Zero errored/empty rows.
- **`extractive_relevance` is the winner: same quality as raw (0.972), prompt cut
  to 28%, 1.36× faster.**
- Decode tok/s is ~88 for all three — compression does **not** change decode
  speed; the win comes entirely from less prompt (prefill) to process. On these
  short-answer tasks total time is prefill-dominated, so cutting the prompt cuts
  wall time directly.

## Where it breaks (real, instructive)

One `multi_fact` task asked which of two options is cheaper over 12 months. With
`semantic_brief` the model answered **"Option A is cheaper"** — the gold answer is
**Option B**. The compressor dropped/blurred the exact monthly figures needed for
the comparison, so the model computed the wrong option (scored q=0.90,
`must_not_include_violation`). This is the known multi-fact + exact-number
failure mode, and the harness **correctly caught and penalized it** rather than
hiding it. `extractive_relevance` (which keeps exact excerpts) got the same task
right — reinforcing the standing recommendation: for numeric/legal exactness,
prefer extractive over abstractive compression.

## Single-document QA (real file, different code path)

```bash
sclab single --runtime ollama --model ornith-moe:latest \
  --document examples/long_contract.txt \
  --question "What is the late payment fee and when does it start accruing?" \
  --compressor extractive_relevance
```

Both raw and compressed returned the identical correct answer
(*"1.5% monthly fee ... after the due date (30 days of receipt)"*); prompt cut
from 295 → 170 tokens (58%), 0.86 s → 0.80 s. Clean.

## Verdict

The compression system works in real use on a 35B MoE: **extractive compression
delivers ~1.3–1.4× faster responses at zero quality loss on long-context QA, and
the harness honestly surfaces the cases where abstractive compression drops a
required fact.** No bugs required fixing in the Ollama path — the one trap that
would bite a reasoning MoE (thinking-dump) is already handled by the default.
