# Phase 6 Memo — Feasibility of an Orthrus-Style Port to Gemma

The checklist's Phase 6 gate: write the feasibility memo *before* attempting any
port. Verdict up front: **the port is a training project, not an engineering
wrapper — do not start it for this lab; use the no-training alternatives that
already work here.**

## What Orthrus actually adds to a base model

From `orthrus-main/src/model_mlx.py` (and the HF checkpoints), Orthrus =
frozen Qwen3 backbone + a **second set of attention projections per layer**
(`q/k/v/o_proj_diff` + `q/k/v_norm_diff`) used by the diffusion view, + a
`mask_token_id` and `block_size` in the config. Both views share one KV cache;
the diffusion view attends over the AR view's cache plus the masked block.
The `_diff` projections are the ~16% trained parameters. **Without training
them, the diffusion view emits noise** — Phase 5.4 already demonstrated the
same lesson for untrained early-exit drafting (0.26–0.41×, a net loss).

## What a Gemma port would require

1. **Architecture mapping.** Orthrus's MLX code assumes Qwen3 specifics:
   per-head q/k RMSNorm, standard global causal attention every layer, SiLU
   gated MLP. Gemma differs in load-bearing ways: alternating sliding-window /
   global attention layers, GeGLU MLP, pre+post RMSNorm placement, a ~256k
   vocab (a mask token must be added or repurposed), and Gemma-specific RoPE
   scaling. Every attention layer needs a `_diff` twin that respects the
   sliding-window pattern — the dual-view KV-sharing trick must be re-derived
   for windowed layers, which is genuinely novel work, not porting.
2. **Training the diffusion view.** ~16% of a Gemma-27/31B-class model is
   4–5B trainable parameters. Orthrus trains on the base model's own
   distribution (the diffusion view must match the AR view's predictions to get
   accepted). That is a multi-GPU training run with data pipeline, eval, and
   the strictly-lossless acceptance criterion to validate — weeks of work and
   rented compute, not something a 64 GB Mac can train.
3. **Gemma 4 specifics are unverified.** The chat notes Google ships official
   **MTP drafters for Gemma 4** with claimed ~3× speedups. If those exist for
   the target checkpoint, they are the trained-drafter route with zero training
   cost to the user — strictly dominating a DIY Orthrus port.

## Cost/benefit against what already works

| route | training | expected gain | status |
|---|---|---|---|
| semantic compression (any model) | none | 1.3–1.9× on long inputs | **working, measured** |
| Orthrus MLX (Qwen3 1.7/4/8B) + router/scheduler | none (pretrained) | 1.5–3.5× structured/reasoning | **working, measured** |
| copy self-speculation (any model) | none | +7% on repetitive output | **working, measured** |
| official Gemma MTP drafters | none | ~3× claimed | verify availability first |
| **DIY Orthrus-Gemma port** | 4–5B params, multi-GPU | ~2–3× decode (if it works) | **not recommended** |

## Recommendation

Do not build Orthrus-Gemma. In order: (1) check whether official Gemma MTP /
speculative drafters exist for the exact target checkpoint and runtime — that
is the same mechanism, pre-trained; (2) meanwhile, run Gemma behind the
compression layer (measured 1.36× on Ornith-35B via Ollama with zero quality
loss) and keep structured-output work on the Orthrus-Qwen3 models where the
2.5–3.5× diffusion win is already real. Revisit a port only if a funded
training budget and a maintainer appetite for the sliding-window dual-view
derivation both materialize.

## Phase 5.5 note (two-machine bucket brigade)

Not testable in this environment (no second Mac present). Design sketch kept:
tiny drafter on machine A streams candidate blocks over Thunderbolt/LAN; the
verifier machine consumes them into the same verify pass the engine already
uses (`CopyProposer`-shaped interface — a network proposer is a drop-in third
proposer type). The engine's proposer abstraction was built so this needs no
decode-loop changes.
