"""sclab runtime adapter for Orthrus dual-view diffusion decoding on MLX.

Implements the :class:`LLMRuntime` protocol so the existing benchmark runner,
compressors, scorer and reporting can drive Orthrus exactly like the ollama
runtime. Exposes the project's north-star metric
(``accepted_tokens_per_verification_pass``) plus peak memory in
``raw_metadata`` so the hard-rule reporting is satisfied.

runtime_options (all optional):
    mode:            "diffusion" (default) | "ar"   -- ar = lossless baseline
    block_size:      int, fixed block for diffusion (default: model config)
    adaptive:        bool, use the acceptance/entropy controller (Phase 2)
    copy:            bool, enable the CopySpec-style copy proposer (Phase 3)
    min_block/max_block: adaptive controller bounds
    enable_thinking: bool, passed to the chat template (default False)

request.model is a short alias resolved to an HF repo id, or a repo id itself.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sclab.runtimes.base import ApproxTokenCounterMixin, GenerationRequest, GenerationResult
from sclab.runtimes.orthrus_engine import (
    BlockPolicy,
    CopyProposer,
    ar_generate,
    load_orthrus,
    orthrus_generate,
    route_mode,
)

MODEL_ALIASES = {
    "orthrus-qwen3-1.7b": "chiennv/Orthrus-Qwen3-1.7B",
    "orthrus-qwen3-4b": "chiennv/Orthrus-Qwen3-4B",
    "orthrus-qwen3-8b": "chiennv/Orthrus-Qwen3-8B",
}


def _resolve_repo(model: str) -> str:
    return MODEL_ALIASES.get(model.lower(), model)


def _peak_memory_bytes() -> Optional[int]:
    try:
        import mlx.core as mx
        for getter in ("get_peak_memory",):
            fn = getattr(mx, getter, None)
            if fn is not None:
                return int(fn())
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "get_peak_memory"):
            return int(metal.get_peak_memory())
    except Exception:
        return None
    return None


def _reset_peak_memory() -> None:
    try:
        import mlx.core as mx
        for name in ("reset_peak_memory",):
            fn = getattr(mx, name, None)
            if fn is not None:
                fn()
                return
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "reset_peak_memory"):
            metal.reset_peak_memory()
    except Exception:
        pass


class OrthrusMLXRuntime(ApproxTokenCounterMixin):
    name = "orthrus-mlx"

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, Any]] = {}

    def is_available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _get_model(self, repo_id: str):
        if repo_id not in self._cache:
            model, tokenizer, _ = load_orthrus(repo_id)
            self._cache[repo_id] = (model, tokenizer)
        return self._cache[repo_id]

    def _encode(self, tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
        """Render the chat template to text, then encode to a clean int list.

        Going through text avoids a transformers/tokenizers quirk where
        ``apply_chat_template(tokenize=True)`` can return Encoding objects
        rather than plain ints for this model.
        """
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tokenize=False,
            )
        except Exception:
            text = prompt
        return list(tokenizer(text, return_tensors=None)["input_ids"])

    def generate(self, request: GenerationRequest) -> GenerationResult:
        opts = request.runtime_options
        repo_id = _resolve_repo(request.model)
        model, tokenizer = self._get_model(repo_id)

        prompt_ids = self._encode(tokenizer, request.prompt, opts.get("enable_thinking", False))
        eos = tokenizer.eos_token_id
        detok = lambda ts: tokenizer.decode(ts)  # noqa: E731

        mode = opts.get("mode", "diffusion")
        route_reason = None
        if mode == "auto":
            mode, route_reason = route_mode(request.prompt)
        policy_mode = opts.get("policy") or ("adaptive" if opts.get("adaptive") else "fixed")
        if policy_mode in ("adaptive", "scheduled"):
            policy = BlockPolicy(
                mode=policy_mode,
                block_size=opts.get("block_size", model.config.block_size),
                min_block=opts.get("min_block", 2),
                max_block=opts.get("max_block", model.config.block_size),
            )
        else:
            policy = BlockPolicy(
                mode="fixed",
                block_size=opts.get("block_size", model.config.block_size),
            )
        copy_proposer = CopyProposer() if opts.get("copy") else None
        prune_tau = opts.get("prune_tau")

        _reset_peak_memory()
        start = time.perf_counter()
        first_token_s: Optional[float] = None
        out_tokens: list[int] = []
        telemetry = None

        if mode == "ar":
            gen = ar_generate(model, prompt_ids, eos, request.max_tokens, request.temperature)
        else:
            gen = orthrus_generate(
                model, prompt_ids, eos, request.max_tokens, request.temperature,
                policy=policy, copy_proposer=copy_proposer, detokenize=detok,
                prune_tau=prune_tau,
            )

        prefill_done_s: Optional[float] = None
        for tok, telemetry in gen:
            if first_token_s is None:
                first_token_s = time.perf_counter() - start
                prefill_done_s = first_token_s
            out_tokens.append(tok)

        total_time = time.perf_counter() - start
        text = tokenizer.decode(out_tokens).strip() if out_tokens else ""
        completion_tokens = len(out_tokens)
        decode_time = (total_time - prefill_done_s) if prefill_done_s is not None else None
        decode_tps = None
        if decode_time and completion_tokens > 1:
            decode_tps = (completion_tokens - 1) / decode_time

        meta: dict[str, Any] = {
            "repo_id": repo_id,
            "mode": mode,
            "requested_mode": opts.get("mode", "diffusion"),
            "route_reason": route_reason,
            "block_policy": policy.mode,
            "block_size": opts.get("block_size", model.config.block_size),
            "prune_tau": prune_tau,
            "copy_proposer": bool(copy_proposer),
            "peak_memory_bytes": _peak_memory_bytes(),
        }
        if telemetry is not None:
            meta.update(telemetry.summary())

        return GenerationResult(
            text=text,
            model=request.model,
            runtime=self.name,
            prompt_tokens=len(prompt_ids),
            completion_tokens=completion_tokens,
            total_time_s=total_time,
            time_to_first_token_s=first_token_s,
            prompt_eval_time_s=prefill_done_s,
            decode_time_s=decode_time,
            decode_tokens_per_s=decode_tps,
            raw_metadata=meta,
        )
