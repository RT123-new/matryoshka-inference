"""Verified speculation for local LLM inference (experimental).

Two lanes, with very different guarantees — do not conflate them:

* **Text-surface mode** (``verify`` + ``loop``): turns an OpenAI-compatible
  ``/v1/completions`` engine that *actually* exposes echoed prompt logprobs (with
  a measurable positional alignment) into a speculative decoder for raw-argmax
  greedy decoding, through its public API. It is **conditional and experimental**:
  :func:`probe_endpoint` must classify an endpoint usable first, and even then it
  proves **surface** identity only, not token-id identity. Distinct token ids that
  decode to the same string, or output that re-tokenizes differently, are beyond
  what it can guarantee.

* **Token-ID mode** (``backend`` + ``token_verify``): verifies drafts on **token
  ids** against a backend's raw argmax, keeping the authoritative context as ids
  across rounds. This is the lane that gives unconditional monolithic-call
  equivalence. The first backend is an in-process ``llama-cpp-python`` adapter
  (``llamacpp_backend``); it is import-guarded and opt-in.

Neither lane is "universal", "lossless" or "faster" without qualification. See
``docs/spec_phase1_results.md`` and ``docs/spec_phase2_results.md`` for exactly
what was demonstrated, on which engines and models, and what remains unproven.
"""

from sclab.spec.backend import (
    DETERMINISTIC_POLICY,
    TOKEN_ID_NONDETERMINISTIC,
    TOKEN_ID_UNAVAILABLE,
    TOKEN_ID_VERIFIED,
    DraftVerification,
    GenerationResult,
    VerificationBackend,
    VerificationCapability,
    logits_argmax,
    policy_is_deterministic,
)
from sclab.spec.loop import SpecStats, spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.token_verify import (
    DraftResolution,
    TokenGeneration,
    TokenSpecStats,
    resolve_draft,
    spec_generate_tokens,
)
from sclab.spec.verify import (
    CAP_AMBIGUOUS_ALIGN,
    CAP_BAD_ALIGN,
    CAP_BAD_SHAPE,
    CAP_BONUS_UNAVAILABLE,
    CAP_CLASSIC,
    CAP_ECHO_IGNORED,
    CAP_ECHO_INCOMPLETE,
    CAP_GENERATED_ONLY,
    CAP_INVALID_OFFSETS,
    CAP_MALFORMED_ARRAYS,
    CAP_NONDETERMINISTIC_POLICY,
    CAP_PARTIAL_COVERAGE,
    CAP_SHIFTED,
    CAP_UNSUPPORTED_OFFSET_UNITS,
    CAP_UNSUPPORTED_TOKEN_IDENTITY,
    EchoToken,
    EndpointCapability,
    Prediction,
    ScoreResult,
    classify_scored_choice,
    generate_burst,
    probe_endpoint,
    score_completion,
)

__all__ = [
    # policy / backend abstraction
    "DETERMINISTIC_POLICY",
    "TOKEN_ID_NONDETERMINISTIC",
    "TOKEN_ID_UNAVAILABLE",
    "TOKEN_ID_VERIFIED",
    "DraftVerification",
    "GenerationResult",
    "VerificationBackend",
    "VerificationCapability",
    "logits_argmax",
    "policy_is_deterministic",
    # token-ID mode
    "DraftResolution",
    "TokenGeneration",
    "TokenSpecStats",
    "resolve_draft",
    "spec_generate_tokens",
    # text-surface mode
    "CAP_AMBIGUOUS_ALIGN",
    "CAP_BAD_ALIGN",
    "CAP_BAD_SHAPE",
    "CAP_BONUS_UNAVAILABLE",
    "CAP_CLASSIC",
    "CAP_ECHO_IGNORED",
    "CAP_ECHO_INCOMPLETE",
    "CAP_GENERATED_ONLY",
    "CAP_INVALID_OFFSETS",
    "CAP_MALFORMED_ARRAYS",
    "CAP_NONDETERMINISTIC_POLICY",
    "CAP_PARTIAL_COVERAGE",
    "CAP_SHIFTED",
    "CAP_UNSUPPORTED_OFFSET_UNITS",
    "CAP_UNSUPPORTED_TOKEN_IDENTITY",
    "EchoToken",
    "EndpointCapability",
    "LookupMemory",
    "Prediction",
    "ScoreResult",
    "SpecStats",
    "classify_scored_choice",
    "generate_burst",
    "probe_endpoint",
    "score_completion",
    "spec_generate",
]
