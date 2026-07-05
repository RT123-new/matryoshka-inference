# Real Hermes Desktop Test + Bug Fixes

A live end-to-end run of the dashboard against **Hermes Agent desktop v0.18**
(Nous Research) driving Ollama's **Ornith-1.0-35B**, plus the bugs it surfaced.

## How Hermes was integrated

Hermes has no "custom OpenAI-compatible endpoint" field in its provider UI — it
stores the model endpoint in `~/.hermes/config.yaml`:

```yaml
model:
  base_url: http://127.0.0.1:11434/v1   # Ollama
  default: ornith-moe:latest
  provider: ollama-launch
```

So the integration is a **transparent proxy**: repoint `base_url` at the proxy
(`sclab hermes-connect`), run the proxy in front of the original endpoint, and
Hermes keeps using the same model while every turn flows through the dashboard.
`hermes-connect --revert` restores the original config.

## Result

Sent "Reply with exactly these three words: proxy test ok" in a fresh Hermes
session. Hermes → proxy (:8977) → Ollama (ornith-moe) → response streamed back
and rendered correctly ("proxy test ok"). The dashboard recorded:

| request | tokens | tok/s | notes |
|---|---:|---:|---|
| chat turn | 1009 | **71.0** | ornith's real 35B throughput (incl. its hidden reasoning) |
| title-gen | 3 | — | Hermes' auto-title request, also captured |

Both appeared in the request feed with prompt previews and `proxy` mode chips.

## Bugs found and fixed

The real run (and inspecting Hermes' request shape) exposed three proxy defects,
all now fixed in `proxy.py` / `server.py`:

1. **Dropped request fields.** The proxy rebuilt a minimal body
   (messages/max_tokens/temperature only), discarding `tools`, `tool_choice`,
   `response_format`, `stop`, `seed`, … — which would **break agent tool
   calling**. Fix: forward the client's *full* body, overriding only the stream
   flags.
2. **Dropped tool-call responses.** The proxy only re-emitted `delta.content`,
   so `tool_calls` never reached the client. Fix: relay the upstream SSE
   **verbatim** and only observe it for telemetry. Verified: a `get_weather`
   tools request round-trips a correct `tool_calls` object.
3. **`/v1/models` returned the wrong list.** It reported the proxy's own served
   name, so a client validating the model against `/v1/models` would fail. Fix:
   proxy `/v1/models` through to the upstream (Hermes now sees the real Ollama
   models).

## Best-UX outcome

- `sclab up` — one command: auto-detect backend (proxy a running Ollama model,
  else Orthrus), start server, open dashboard.
- `sclab hermes-connect` / `--revert` — one-command, reversible Hermes wiring.
- Transparent, faithful proxy — any model, any OpenAI-compatible client, tools
  and streaming preserved.

## Safety note

The test left Hermes' config **reverted to its original** endpoint, so Hermes is
never dependent on an ephemeral proxy. Re-connect on demand with
`sclab hermes-connect` while your own `sclab serve` proxy is running.
