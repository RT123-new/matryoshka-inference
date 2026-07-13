# Matryoshka Inference

**Lossless local LLM acceleration on Apple Silicon** — semantic prompt
compression (works with *any* model, including Ollama) composed with verified
speculative decoding (Orthrus dual-view diffusion on MLX), a per-request mode
router, and a DSpark-style speculation scheduler. Every accelerated path is
verified by the exact autoregressive pass, so output quality is preserved by
construction — and every claim in this README comes from a benchmark you can
re-run from the CLI.

**Headline (measured, 30-task long-context QA, M4 Max, Orthrus-Qwen3-4B):**

| | plain decode | accelerated decode |
|---|---|---|
| raw prompt | 1.52 s (baseline) | 1.02 s |
| compressed prompt | 0.90 s | **0.55 s → 2.79× faster** |

Answer quality with compression was *equal or better* than raw (0.959 vs 0.944 —
less distractor text). Zero errors in 180 result rows.

## How it works

Two independent levers that attack different bottlenecks, so they multiply:

```
long input ──[semantic compressor]──► short prompt      (attacks PREFILL, any model)
                                          │
                                 [mode router: per request]
                                          │
                    structured/reasoning ─┴─ free-form prose
                             │                    │
              [diffusion draft → AR verify]   [plain AR]
              (attacks DECODE; scheduler       (drafting loses here —
               backs off if drafts fail)        measured, not assumed)
```

1. **Semantic compression** (`extractive_relevance`, `semantic_brief`, …) cuts
   long prompts to 25–40% of their tokens while preserving the facts needed to
   answer. Works with Ollama, llama.cpp, MLX — anything.
2. **Orthrus dual-view diffusion decoding** (vendored, MIT) proposes multi-token
   blocks that the same model's AR pass verifies — 2.2–3.5× on structured /
   repetitive / reasoning output, scaling with model size (8B > 4B > 1.7B).
3. **Router + scheduler** keep it safe: prompts routed to the winning mode per
   request; inside diffusion mode, a DSpark-style scheduler drops to a plain-AR
   lane when measured acceptance collapses (worst case 0.90× instead of 0.55×).
4. A **copy proposer** (CopySpec-style) serves repeated spans (JSON, templates)
   without even paying the draft pass.

## Install

```bash
git clone https://github.com/RT123-new/matryoshka-inference
cd matryoshka-inference
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[orthrus,dev]"     # [orthrus] = MLX decode path (Apple Silicon)
pip install -e .                    # compression-only (any platform / runtime)
pytest -q                           # full test suite, no GPU required
```

Orthrus checkpoints download automatically from Hugging Face on first use
(`chiennv/Orthrus-Qwen3-1.7B` / `-4B` / `-8B`; ~4/8/16 GB).

## One command

```bash
./quickstart.sh
```

Creates a venv, installs the package, starts the server, and opens the live
dashboard in your browser. With no arguments it **auto-detects**: if Ollama is
running it proxies your first Ollama model (model-agnostic mode); otherwise it
loads the accelerated Orthrus-Qwen3-4B. Already installed? Just run `sclab up`.

## Quickstart

### Model-agnostic: works with ANY model (Ollama, LM Studio, llama.cpp, vLLM)

Put the dashboard in front of whatever you already run. It measures real
tokens/sec token-by-token, so the dashboard updates correctly regardless of the
model:

```bash
# proxy a running Ollama model — the dashboard now tracks it live
sclab serve --backend proxy --upstream http://localhost:11434/v1 --model gemma4:latest
# zero-flag version: defaults the upstream to Ollama and auto-picks its first model
sclab serve --backend proxy
# LM Studio:  --upstream http://localhost:1234/v1
# any OpenAI-compatible API:  --upstream <base-url> --api-key <key>
```

Point your client (or Hermes) at `http://127.0.0.1:8977/v1` and open
`http://127.0.0.1:8977/dashboard`. In proxy mode the Orthrus-only metrics
(accepted/pass, draft acceptance) show `—`; tokens/sec, throughput, token count
and the request feed are live for the external model. Chat completions are
observed for telemetry; every other `/v1` endpoint the upstream serves
(embeddings, legacy completions, ...) passes through untouched.

### Accelerated: Orthrus dual-view diffusion (MLX, Apple Silicon)

```bash
sclab serve --model orthrus-qwen3-4b --port 8977      # or: sclab up --model orthrus-qwen3-4b
```

Open **http://127.0.0.1:8977/dashboard** — a real-time view of the engine
working: a live pipeline animation (Prompt → Router → Draft/Verify → Stream),
tokens/sec, accepted-tokens-per-verification-pass, draft acceptance %, live
speedup vs AR, a throughput chart, a token-source breakdown, and a request feed.
Every request from any client (including Hermes) lights it up. It also has a
built-in **playground** so you can watch it work without any other app. The
dashboard is self-contained (no external assets) and safe to embed in a desktop
webview.

Then from anything that speaks the OpenAI API:

```bash
curl http://127.0.0.1:8977/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "orthrus-qwen3-4b",
  "messages": [{"role": "user", "content": "Output a JSON array of 5 user objects."}],
  "max_tokens": 300, "stream": true
}'
```

The response's `sclab` field reports decode mode, accepted-tokens-per-pass and
source mix, so you can see the acceleration working. `--mode auto` (default)
routes each request; force `--mode diffusion` or `--mode ar` to pin it.

### Compression with Ollama (any model, today)

Compress a long document before sending it — measured 1.36× faster at equal
quality on a 35B MoE, and up to 1.9× on long contexts:

```bash
Q="What is the late payment fee and when does it start accruing?"
sclab compress --document contract.txt --question "$Q" --stats \
  | ollama run gemma4:latest "Answer from this source:\n$(cat -)\n\nQuestion: $Q"
```

Or run the built-in benchmark harness against your own Ollama model:

```bash
sclab single --runtime ollama --model gemma4:latest \
  --document examples/long_contract.txt \
  --question "$Q" --compressor extractive_relevance
```

### A/B: measure Matryoshka off vs on, same prompts

```bash
sclab ab --open                                        # compression A/B on your current Hermes model
sclab ab --backend orthrus --model orthrus-qwen3-4b --open   # AR-vs-diffusion acceleration A/B
```

Runs the same diverse prompts twice (baseline vs Matryoshka) with identical
measurements and writes a side-by-side HTML report (both outputs, tok/s,
accepted/pass, quality). Findings on the bundled prompts: compression on a stock
Ollama model is a situational trade (1.13×, some quality loss on multi-fact),
while **diffusion decoding on an Orthrus model is a near-lossless 1.71× (up to
2.28×) with 7/8 token-identical outputs** — see [docs/ab_findings.md](docs/ab_findings.md).

### Reproduce the findings

```bash
sclab benchmark --runtime orthrus-mlx --model orthrus-qwen3-4b \
  --dataset data/tasks/synthetic_long.jsonl \
  --compressors raw,semantic_brief,extractive_relevance \
  --max-tokens 160 --runtime-options '{"mode":"diffusion","block_size":16}' \
  --out runs/my_repro
```

Every run writes raw JSONL + a Markdown report, always against a raw baseline.

## Using it with popular tools

### Hermes Agent (desktop) — verified

Hermes talks to a local model via an OpenAI-compatible base URL (its default is
Ollama at `http://127.0.0.1:11434/v1`). The cleanest integration is to slot the
**transparent proxy** between Hermes and its existing endpoint — Hermes keeps
using the exact same model, and every agent turn (chat, tool calls, title
generation) shows up in the dashboard. One command does it:

```bash
sclab hermes-connect          # repoints Hermes' base_url at the proxy (backs up config)
# it prints the proxy command to run, e.g.:
sclab serve --backend proxy --upstream http://127.0.0.1:11434/v1 --port 8977 --open
# then fully quit & reopen Hermes. Disconnect anytime:
sclab hermes-connect --revert  # restores the original config; restart Hermes
```

That's it — chat in Hermes and watch `http://127.0.0.1:8977/dashboard` light up
in real time. Tool calls and streaming pass through untouched (the proxy relays
the upstream response verbatim). **This was tested end-to-end** against Hermes
v0.18 driving Ollama's Ornith-1.0-35B: the dashboard captured each turn at the
model's real ~71 tok/s.

> Keep the proxy running while connected — if it stops, Hermes can't reach the
> endpoint until you `--revert`. Hermes has no plugin API to embed a panel inside
> its own window, so the dashboard is its own browser tab (`--open` launches it).

Prefer the accelerated Orthrus model in Hermes instead of proxying? Point
`hermes-connect` at an Orthrus server (`sclab serve --model orthrus-qwen3-4b`)
and set Hermes' model to `orthrus-qwen3-4b` — you then get the acceleration
metrics too. Note a 1.7–4B model is a weak *agent*; the transparent proxy in
front of your usual model is the better everyday setup.

Hermes' `/models` probe is supported, so the model appears in its picker.
Config lands in `~/.hermes/config.yaml` if you prefer editing it directly.

**Watch it while you chat:** keep `http://127.0.0.1:8977/dashboard` open in a
browser tab (or a side webview) next to Hermes — as you chat, each turn flows
through the server and the dashboard's pipeline, tok/s, and acceptance metrics
update live. This is the "shows exactly how it works" view: you can see which
prompts route to diffusion vs AR and how many tokens each verification pass
accepts, in real time.

### Ollama

Two ways to benefit:

- **Compression layer** (no server needed): use `sclab compress` / `sclab
  single` / `sclab benchmark --runtime ollama` as above. Works with every model
  in your `ollama list`.
- **Tip for reasoning models** (Gemma, Ornith, Qwen): the harness defaults to
  `think=false` because thinking-mode models otherwise burn the token budget on
  hidden reasoning and return empty answers — we hit this on both `gemma4` and
  `ornith-moe`.

### Open WebUI / LM Studio / Jan / anything OpenAI-compatible

Point the app's OpenAI-compatible provider at `http://127.0.0.1:8977/v1` with
any API key. Streaming (SSE) and non-streaming are both supported.

### Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8977/v1", api_key="local")
r = client.chat.completions.create(
    model="orthrus-qwen3-4b",
    messages=[{"role": "user", "content": "Solve step by step: 3x+7=28"}],
)
print(r.choices[0].message.content)
```

## Findings (all measured on an M4 Max / 64 GB, greedy decoding)

The full story is in [`docs/final_report.md`](docs/final_report.md); raw data in
[`results/`](results/). The short version:

**What works**

- **Compression × diffusion multiply: 2.79×** end-to-end on long-context QA at
  equal-or-better quality ([docs](docs/orthrus_phase4_results.md)).
- **The speedup grows with model size** — json workload: 2.06× (1.7B) → 2.26×
  (4B) → **3.49× (8B)** — because bigger models are memory-bandwidth-bound, so
  each avoided sequential pass is worth more ([docs](docs/orthrus_phase1_results.md)).
- **Per-request routing beats both pure strategies** (1.48× vs always-AR, 1.33×
  vs always-diffusion on mixed workloads) ([docs](docs/orthrus_phase5_results.md)).
- **A DSpark-style speculation scheduler makes always-on diffusion safe**: worst
  case improves 0.55× → 0.90× while winning workloads keep their full speedup
  ([docs](docs/dspark_integration_results.md)).
- **Copy-speculation** serves 40% of tokens on repetitive JSON at +7% throughput.
- Compression alone: **1.36× at identical quality on a 35B MoE via Ollama**
  ([docs](docs/ornith_moe_real_test.md)).

**What doesn't (negative results, kept visible)**

- **Layer-skip self-speculation on a stock model: net loss** (0.26–0.41×) —
  untrained early-exit logits don't draft ([docs](docs/orthrus_phase5_4_layerskip.md)).
- **DSpark-style confidence pruning**: replicates the acceptance effect
  (40→88%) but is a wall-clock wash single-stream on Apple Silicon — verify
  width is nearly free there; the mechanism pays on batched serving.
- **Outline-conditioned drafting**: +0.1pt acceptance on prose — no effect.
- **Thinking spans draft like prose**, not boilerplate — route thinking models
  to plain AR.
- **Abstractive compression can be confidently wrong** on multi-fact numeric
  comparisons; prefer `extractive_relevance` when exact numbers matter — and
  gold-blind fault-in triggers can't catch it without logprobs.

**Losslessness, honestly**: accelerated output is verified by the exact AR pass.
Divergences vs token-by-token decoding occur only at floating-point near-ties
(measured logit gap 0.125/32) where the model is indifferent — documented in
[`docs/orthrus_phase1_results.md`](docs/orthrus_phase1_results.md).

## Repo layout

```
src/sclab/                 the package (compressors, runtimes, benchmarks, server)
src/sclab/runtimes/orthrus_engine.py   instrumented decode loop: proposers,
                                       scheduler, pruning, router, telemetry
src/sclab/vendor/orthrus/  Orthrus MLX architecture (MIT, Chien Nguyen)
docs/                      all findings write-ups (positive AND negative)
results/                   raw JSONL from every headline benchmark
data/tasks/                benchmark datasets (short + long-context)
tests/                     test suite (incl. live HTTP proxy tests), no GPU required
```

## Credits

- [Orthrus](https://github.com/chiennv2000/orthrus) by Chien Nguyen — the
  dual-view diffusion architecture and pretrained checkpoints (MIT).
- DeepSeek's DSpark for the confidence-scheduled-verification idea we adapted
  (and honestly benchmarked) for single-stream Apple Silicon.
- CopySpec, Draft & Verify, Medusa/EAGLE, and the MTP literature for the
  speculative-decoding foundations.

MIT licensed. Benchmarks were run on one machine (M4 Max, 64 GB); your numbers
will vary — the harness exists so you can measure, not trust.
