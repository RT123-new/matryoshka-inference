"""In-process ``llama-cpp-python`` adapter — the first token-ID-capable backend.

Phase 1 established that ``llama-cpp-python``'s *HTTP* server exposes echoed
prompt logprobs (shifted by one), but only as decoded token *surfaces* — never
token ids — so text-mode verification can only prove surface identity. The
in-process ``llama_cpp.Llama`` API does expose the real logits, so this adapter
verifies drafts on **token ids** against the target's raw argmax, closing that
gap without patching upstream ``llama.cpp``.

It is deliberately narrow and honest about scope:

* It requires ``logits_all=True`` so every draft position has real logits; if the
  model was loaded without it, :meth:`capability` reports the lane unusable and
  the loop falls back to plain generation.
* It enforces raw-argmax greedy with a lowest-id tie-break, matching
  ``backend.logits_argmax`` and llama.cpp's own greedy sampler, so a fake backend
  and this one agree bit-for-bit at ties.
* It makes **no** claim of universality. It is one backend, for one engine's
  in-process API. Other engines need their own adapters.

The import of ``llama_cpp`` is lazy and guarded: importing this module never
requires the package, so ``sclab`` still imports on a machine with no engine.
This adapter is exercised only by opt-in real-engine tests when a local GGUF is
present (``SCLAB_SPEC_TEST_GGUF``); it is not run in normal CI.
"""

from __future__ import annotations

from typing import Any

from sclab.spec.backend import (
    DETERMINISTIC_POLICY,
    TOKEN_ID_UNAVAILABLE,
    TOKEN_ID_VERIFIED,
    DraftVerification,
    GenerationResult,
    VerificationCapability,
    logits_argmax,
)


class LlamaCppBackend:
    """A :class:`~sclab.spec.backend.VerificationBackend` over ``llama_cpp.Llama``.

    Construct with an already-loaded ``Llama`` instance, or use
    :meth:`from_model_path`. The instance **must** have been created with
    ``logits_all=True`` for the verify lane to be usable.
    """

    def __init__(self, llm: Any, logits_all: bool = True) -> None:
        self._llm = llm
        self._logits_all = logits_all
        try:
            self._eos = int(llm.token_eos())
        except Exception:  # pragma: no cover - defensive; older/newer API shapes
            self._eos = None

    @classmethod
    def from_model_path(cls, model_path: str, n_ctx: int = 8192,
                        n_gpu_layers: int = 0, **kwargs: Any) -> LlamaCppBackend:
        """Load a GGUF with the settings token-ID verification requires.

        ``logits_all=True`` is forced (a warning-free necessity, not an option);
        ``n_ctx`` should exceed prompt + draft + budget. No weights are fetched
        here — the path must already exist locally.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "llama-cpp-python is not installed; token-ID mode needs the "
                "in-process llama_cpp.Llama API (pip install llama-cpp-python)."
            ) from exc
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                    logits_all=True, verbose=False, **kwargs)
        return cls(llm, logits_all=True)

    # -- capability -------------------------------------------------------- #
    def capability(self) -> VerificationCapability:
        if not self._logits_all:
            return VerificationCapability(
                mode=TOKEN_ID_UNAVAILABLE, deterministic=False,
                detail="Llama loaded without logits_all=True; per-position draft "
                       "logits are unavailable, cannot verify token ids.")
        return VerificationCapability(
            mode=TOKEN_ID_VERIFIED, deterministic=True, supports_bonus=True,
            eos_token_id=self._eos, policy=dict(DETERMINISTIC_POLICY),
            detail="in-process llama_cpp.Llama, raw-argmax greedy over real logits")

    # -- tokenizer --------------------------------------------------------- #
    def encode_context(self, text: str) -> list[int]:
        # add_bos mirrors how the engine tokenizes a fresh prompt; special=False
        # keeps user text literal (no accidental control-token injection).
        toks = self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        return [int(t) for t in toks]

    def decode_tokens(self, token_ids: list[int]) -> bytes:
        if not token_ids:
            return b""
        return bytes(self._llm.detokenize([int(t) for t in token_ids]))

    # -- logits ------------------------------------------------------------ #
    def _scores_row(self, index: int):
        """One logits row from the current eval, tolerant of API attribute drift."""
        scores = getattr(self._llm, "scores", None)
        if scores is None:
            scores = self._llm._scores  # noqa: SLF001 - fallback for API attribute drift
        return scores[index]

    def generate_plain(self, context_ids: list[int], max_tokens: int) -> GenerationResult:
        if max_tokens <= 0:
            return GenerationResult(token_ids=[], finish_reason="length")
        try:
            llm = self._llm
            llm.reset()
            llm.eval([int(t) for t in context_ids])
            out: list[int] = []
            finish = "length"
            for _ in range(max_tokens):
                nxt = logits_argmax(self._scores_row(llm.n_tokens - 1))
                if nxt == self._eos:
                    finish = "stop"
                    break
                out.append(nxt)
                llm.eval([nxt])
            return GenerationResult(token_ids=out, finish_reason=finish)
        except Exception as exc:  # pragma: no cover - engine/runtime dependent
            return GenerationResult(error=f"llama_cpp generate_plain failed: {exc}")

    def verify_draft(self, context_ids: list[int], draft_ids: list[int]) -> DraftVerification:
        if not context_ids:
            return DraftVerification(error="empty context (need at least BOS)")
        try:
            llm = self._llm
            seq = [int(t) for t in context_ids] + [int(t) for t in draft_ids]
            llm.reset()
            llm.eval(seq)
            base = len(context_ids)
            # logits[i] predicts token i+1, so the argmax for absolute position
            # base+j (draft position j) is read from row base-1+j; one extra row
            # (j == len(draft)) is the bonus prediction after the whole draft.
            predicted = [
                logits_argmax(self._scores_row(base - 1 + j))
                for j in range(len(draft_ids) + 1)
            ]
            return DraftVerification(predicted_ids=predicted)
        except Exception as exc:  # pragma: no cover - engine/runtime dependent
            return DraftVerification(error=f"llama_cpp verify_draft failed: {exc}")
