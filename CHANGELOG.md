# Changelog

## Unreleased

### Changed — token-ID verified speculation + a hardened text probe (spec Phase 2)

- New `VerificationBackend` abstraction (`sclab.spec.backend`,
  `sclab.spec.token_verify`): verifies drafts on **token ids** against a
  backend's raw argmax and keeps the authoritative context as ids across rounds,
  so output is **id- and byte-identical** to a single plain call — the
  unconditional equivalence text-surface mode cannot give. First backend: an
  opt-in in-process `llama-cpp-python` adapter (`sclab.spec.llamacpp_backend`).
  Proven to the byte against deterministic fixtures; **no trained model was
  reachable**, so trained-model equivalence is set up (opt-in) but not run.
- **Hardened the text-surface probe.** An endpoint is usable only when every
  invariant holds: complete echo, compatible/finite logprob arrays, monotonic
  non-overlapping **code-point** offsets tiling the continuation, a single
  100%-verifying alignment shift, and an unambiguous bonus. Byte offsets,
  ambiguous alignment, byte-fallback surfaces and non-greedy policies are
  rejected; `usable` now requires `offsets_ok` **and** `bonus_ok`.
  `spec_generate(..., capability=None)` runs plain generation — no default
  assumes an alignment. The whitespace token-budget estimate is removed, and
  verify/burst failures degrade to plain generation instead of truncating.
- **Benchmark honesty:** speed is gated on the byte-identity correctness result;
  timing validates every call, alternates order, separates cold/warm/cached
  regimes, reports median with min/max and request counts, and saves
  machine-readable results (`results/spec_phase2/`). Text-surface mode is now
  documented as **conditional and experimental**; see
  [`docs/spec_phase2_results.md`](docs/spec_phase2_results.md).

### Added — API-level verified speculation, text-surface lane (experimental)

_Phase 1 correction: what Phase 0 called "universal, byte-identical on any
OpenAI-compatible engine" was narrowed — see
[`docs/spec_phase1_results.md`](docs/spec_phase1_results.md). It works only on an
endpoint that passes a behavioural probe (`llama-cpp-python`, which needed a +1
alignment fix; **not** native `llama-server`), proves surface identity only, and
showed no speedup on the CPU test rig._

- New `sclab.spec` package: lossless speculative decoding through *any*
  OpenAI-compatible `/v1/completions` engine, with **no draft model, no engine
  patch, and no special checkpoint**. Drafts come from a zero-cost text lookup
  memory (`LookupMemory`); verification is one `echo`+`logprobs` scoring
  round-trip that proves each draft token equals the engine's own greedy
  choice. Every emitted token is the engine's greedy token by construction, so
  output is byte-identical to plain decoding.
- `sclab spec-bench` CLI: baseline-vs-spec on a real engine, or `--sim` for an
  instant local demo with no model. `--cost-probe` measures the engine's
  scoring-vs-decoding breakeven (≈1.0 on memory-bandwidth-bound decoders — the
  physics that makes the approach win).
- Deterministic sim engine (`sclab.spec.sim`) with exact `echo`/`logprobs`
  semantics and an optional latency model, so losslessness is proven in CI
  with no GPU. Tests grew 56 → 105, including a byte-identity matrix over
  workloads, draft alignments, and token budgets.
- Full design + roadmap in [`HANDOFF.md`](HANDOFF.md): the path from this
  prototype to a universal, model-agnostic, lossless decode accelerator folded
  into the transparent proxy.

## 0.2.0 — 2026-07-13

Robustness and correctness release for the server, proxy, and telemetry.
No benchmark methodology changes; all published numbers still hold.

### Fixed

- **Server:** responses could be wiped or crash when the tokenizer's
  `eos_token` was empty or `None` (the old suffix-strip ran `text[:0]` /
  `len(None)`); EOS is now excluded by token id instead of string surgery.
- **Server:** OpenAI content-part messages
  (`content: [{"type": "text", ...}]`) crashed the Orthrus backend; both
  backends now flatten them.
- **Server:** the dashboard playground sent `model: "local"`, which proxy
  mode forwarded verbatim and upstreams rejected; the placeholder now maps to
  the served model (real client model names still pass through untouched).
- **Server:** concurrent requests could interleave MLX decode state —
  `ThreadingHTTPServer` handles each request on its own thread. Orthrus
  generation is now serialised by a lock; proxy requests stay concurrent.
- **Telemetry:** overlapping requests trampled a single shared "live" slot,
  attributing one request's tokens/timings to another. Sessions are now keyed
  by request id; `accepted_from_draft` uses the real accepted count instead of
  a rate×passes approximation.
- **Engine:** diffusion generation could overshoot `max_tokens` by one; block
  size is now capped at the remaining budget.
- **Ollama runtime:** mid-stream `{"error": ...}` events (model OOM, ...) were
  swallowed and returned as a silent empty answer; they are now surfaced in
  `raw_metadata["error"]`, and malformed stream lines are skipped.
- **Proxy streaming:** the usage chunk injected for telemetry leaked to
  clients that never requested `stream_options.include_usage`; it is now
  withheld unless the client asked (and exact token counts still feed the
  dashboard).
- `sclab ab --open` used the macOS-only `open` command; now uses the standard
  browser opener on every platform.

### Added

- Proxy mode relays **any other `/v1` endpoint** (embeddings, legacy
  completions, ...) to the upstream byte-for-byte, so a client pointed at the
  proxy never hits a 404 the real endpoint would have answered.
- CORS headers + `OPTIONS` preflight, so browser-based OpenAI clients can call
  the server directly.
- OpenAI-spec `finish_reason` (`"stop"` vs `"length"`) and a final usage chunk
  when a streaming client sends `stream_options.include_usage`.
- `sclab serve --backend proxy` with no flags: defaults the upstream to local
  Ollama and auto-detects the first model via the upstream's `/models`.
- `sclab --version`, friendly errors for missing document files, bad
  `--budgets` / `--runtime-options`, a busy port, and running the Orthrus
  backend without MLX installed.
- Mid-generation failures now return a 500 (non-streaming) or an SSE error
  frame + `[DONE]` (streaming) instead of killing the connection silently.
- Dashboard: shows in-flight request count, escapes upstream-controlled model
  names, renders `—` for proxy rows' Orthrus-only columns, and surfaces
  streamed error frames in the playground.
- Test suite grew from 30 to 56 tests, including live-HTTP proxy integration
  tests (fake upstream + real handler on loopback ports) — still no GPU
  required. Ruff lint config + CI workflow.

### Changed

- Dropped unused `pydantic`, `rich`, and `numpy` dependencies (numpy still
  arrives transitively via scikit-learn); leaner default install.
