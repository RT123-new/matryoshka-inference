# Next Steps Results

This note records the follow-up work after the initial setup.

## What Was Run

1. Verified the project and tests.
2. Ran a real Ollama benchmark on the short built-in dataset.
3. Found and fixed an Ollama/Gemma issue where `gemma4:latest` spent the token budget on hidden thinking and returned empty visible answers.
4. Tuned deterministic scoring so correct paraphrases with the required numbers, dates, names, or commands are not marked as poor answers just because wording differs.
5. Generated a long-context dataset with distractor paragraphs.
6. Tuned compressors so generated distractors do not dominate semantic briefs or fact tables.
7. Ran a full 30-task long-context benchmark with `gemma4:latest`.
8. Ran a real single-document check on `examples/long_contract.txt`.

## Main Evidence

Full run:

```text
runs/gemma4_latest_long_30_tuned/
```

The strongest result was on long-context tasks:

| compressor | avg quality | avg prompt size | avg speed factor | pass rate |
|---|---:|---:|---:|---:|
| semantic_brief | 0.977 | 40.9% of raw | 1.54x faster | 96.7% |
| extractive_relevance | 0.944 | 20.9% of raw | 1.90x faster | 96.7% |
| hybrid_brief_excerpts | 0.960 | 50.2% of raw | 1.29x faster | 76.7% |
| fact_table | 0.910 | 28.1% of raw | 1.79x faster | 86.7% |
| raw | 0.949 | 100% | baseline | 0.0% |
| gzip_b64_control | 0.133 | 122.3% of raw | misleading | 0.0% |

The gzip/base64 control failed as expected, which supports the point that file-style compression is not semantic compression.

## Plain-English Takeaway

Compression helped when the source document was long.

The best options were:

- `semantic_brief`: best answer quality in this run.
- `extractive_relevance`: fastest and smallest prompt while still passing most tasks.

Compression did not help on short documents. For short inputs, the extra compressed-source instructions can make the prompt bigger and slower than raw text.

## Where It Still Breaks

The weakest categories were:

- multi-fact synthesis
- contradiction detection
- exact number/date tasks when a compressor drops one required detail

For legal, financial, or technical exactness, prefer compressors that preserve exact raw excerpts.

## Current Recommendation

Continue prompt-layer semantic compression experiments for long-context factual QA.

Do not claim this proves model-internal compressed reasoning. It shows that an outer prompt-layer compression step can reduce local inference time on long inputs while preserving answer quality in many cases.

Next best research step:

```bash
python -m sclab ablate \
  --runtime ollama \
  --model gemma4:latest \
  --dataset data/tasks/synthetic_long.jsonl \
  --compressor semantic_brief \
  --budgets 0.2,0.4,0.6,0.8 \
  --max-tasks 30 \
  --out runs/gemma4_latest_semantic_ablation
```
