#!/usr/bin/env python3
"""Build the synthetic GGUF fixtures used in Spec Phase 1 (docs/spec_phase1_results.md).

Hugging Face was blocked by egress policy on the Phase 1 rig, so no trained GGUF
could be downloaded. These fixtures let a *real* llama.cpp / llama-cpp-python
engine run with a *real* GPT-2 BPE tokenizer and real greedy determinism, which
is what the capability probe, the alignment diagnostic, and the losslessness
gate need. They are NOT trained language models — see the doc for the caveats.

Requires the `gguf` writer and a vocab-only GGUF for the tokenizer. Both ship
with llama.cpp:

    python scripts/spec_build_fixtures.py \
        --vocab /path/to/llama.cpp/models/ggml-vocab-gpt-2.gguf \
        --out-dir /path/to/models [--gguf-py /path/to/llama.cpp/gguf-py]

Then serve one with, e.g.:

    python -m llama_cpp.server --model models/tiny-cycle-gpt2.gguf \
        --n_ctx 8192 --logits_all true --port 8081
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

N_LAYER, N_EMBD, N_HEAD, N_FF, N_CTX = 4, 128, 4, 256, 8192
HEAD_DIM = N_EMBD // N_HEAD


def _load_gguf(gguf_py: str | None):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
    return GGUFReader, GGUFWriter, GGUFValueType


def _base_writer(GGUFWriter, GGUFValueType, reader, out: Path, name: str):
    wr = GGUFWriter(str(out), "llama")
    wr.add_name(name)
    wr.add_context_length(N_CTX)
    wr.add_embedding_length(N_EMBD)
    wr.add_block_count(N_LAYER)
    wr.add_feed_forward_length(N_FF)
    wr.add_head_count(N_HEAD)
    wr.add_head_count_kv(N_HEAD)
    wr.add_rope_dimension_count(HEAD_DIM)
    wr.add_layer_norm_rms_eps(1e-5)
    wr.add_rope_freq_base(10000.0)
    wr.add_file_type(0)  # ALL_F32
    for field in reader.fields.values():   # copy the REAL tokenizer verbatim
        if field.name.startswith("tokenizer."):
            vt = field.types[0]
            sub = field.types[-1] if vt == GGUFValueType.ARRAY else None
            wr.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    return wr


def _write(wr):
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()


def build_random(GGUFReader, GGUFWriter, GGUFValueType, vocab: str, out: Path):
    """Random weights: pathological (near-ties, degenerate token, seams). A warning, not a model."""
    r = GGUFReader(vocab)
    n_vocab = len(r.get_field("tokenizer.ggml.tokens").data)
    rng = np.random.default_rng(0)
    w = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)  # noqa: E731
    ones = lambda *s: np.ones(s, dtype=np.float32)                     # noqa: E731
    wr = _base_writer(GGUFWriter, GGUFValueType, r, out, "tiny-llama-gpt2-untrained")
    wr.add_tensor("token_embd.weight", w(n_vocab, N_EMBD))
    wr.add_tensor("output_norm.weight", ones(N_EMBD))
    wr.add_tensor("output.weight", w(n_vocab, N_EMBD))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        wr.add_tensor(p + "attn_norm.weight", ones(N_EMBD))
        wr.add_tensor(p + "attn_q.weight", w(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_k.weight", w(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_v.weight", w(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_output.weight", w(N_EMBD, N_EMBD))
        wr.add_tensor(p + "ffn_norm.weight", ones(N_EMBD))
        wr.add_tensor(p + "ffn_gate.weight", w(N_FF, N_EMBD))
        wr.add_tensor(p + "ffn_up.weight", w(N_FF, N_EMBD))
        wr.add_tensor(p + "ffn_down.weight", w(N_EMBD, N_FF))
    _write(wr)
    print(f"wrote {out}: random weights, vocab={n_vocab}")


def build_cycle(GGUFReader, GGUFWriter, GGUFValueType, vocab: str, out: Path, cycle: int = 64):
    """Confident, canonical, periodic output: a legitimate real-engine integration fixture.

    Near-zero attention/FFN pass token_embd[current] to the head; the head maps
    each canonical leading-space word token to the next in a fixed cycle, so the
    argmax is confident (prefill == decode) and every emitted token re-tokenizes
    to itself. It is NOT a language model; its acceptance is periodicity, not a workload.
    """
    r = GGUFReader(vocab)
    tok = r.get_field("tokenizer.ggml.tokens")
    n_vocab = len(tok.data)
    surfaces = [tok.contents(i) for i in range(n_vocab)]
    ids = [i for i, s in enumerate(surfaces)
           if s.startswith("Ġ") and s[1:].isalpha() and s[1:].islower()
           and 3 <= len(s) - 1 <= 7][:cycle]
    if len(ids) < cycle:
        raise SystemExit(f"only found {len(ids)} canonical cycle tokens (< {cycle})")
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((n_vocab, N_EMBD)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    out_w = np.zeros((n_vocab, N_EMBD), dtype=np.float32)
    for k, cid in enumerate(ids):
        out_w[ids[(k + 1) % cycle]] = 8.0 * emb[cid]   # next(cid) aligned with emb[cid]
    tiny = lambda *s: (rng.standard_normal(s) * 1e-4).astype(np.float32)  # noqa: E731
    ones = lambda *s: np.ones(s, dtype=np.float32)                        # noqa: E731
    wr = _base_writer(GGUFWriter, GGUFValueType, r, out, "tiny-cycle-gpt2-confident-canonical")
    wr.add_tensor("token_embd.weight", emb)
    wr.add_tensor("output_norm.weight", ones(N_EMBD))
    wr.add_tensor("output.weight", out_w)
    for i in range(N_LAYER):
        p = f"blk.{i}."
        wr.add_tensor(p + "attn_norm.weight", ones(N_EMBD))
        wr.add_tensor(p + "attn_q.weight", tiny(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_k.weight", tiny(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_v.weight", tiny(N_EMBD, N_EMBD))
        wr.add_tensor(p + "attn_output.weight", tiny(N_EMBD, N_EMBD))
        wr.add_tensor(p + "ffn_norm.weight", ones(N_EMBD))
        wr.add_tensor(p + "ffn_gate.weight", tiny(N_FF, N_EMBD))
        wr.add_tensor(p + "ffn_up.weight", tiny(N_FF, N_EMBD))
        wr.add_tensor(p + "ffn_down.weight", tiny(N_EMBD, N_FF))
    _write(wr)
    print(f"wrote {out}: cycle={cycle}, vocab={n_vocab}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", required=True, help="a vocab-only GGUF, e.g. ggml-vocab-gpt-2.gguf")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--gguf-py", default=None, help="path to llama.cpp/gguf-py if gguf isn't installed")
    ap.add_argument("--cycle", type=int, default=64)
    args = ap.parse_args()
    GGUFReader, GGUFWriter, GGUFValueType = _load_gguf(args.gguf_py)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_random(GGUFReader, GGUFWriter, GGUFValueType, args.vocab, out / "tiny-llama-gpt2.gguf")
    build_cycle(GGUFReader, GGUFWriter, GGUFValueType, args.vocab, out / "tiny-cycle-gpt2.gguf", args.cycle)


if __name__ == "__main__":
    main()
