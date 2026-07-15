"""API-level verified speculation (experimental).

Turns an OpenAI-compatible engine that *actually* exposes completion scoring
(``echo`` + prompt ``logprobs`` on ``/v1/completions``, with a measurable
positional alignment) into a lossless speculative decoder for raw-argmax greedy
decoding — through its public API, with no engine patch, no special checkpoint,
and no draft model. Not all engines qualify: use :func:`probe_endpoint` to
classify one before speculating, and fall back to plain generation otherwise.
See ``docs/spec_phase1_results.md`` for what survived real-engine testing.
"""

from sclab.spec.loop import SpecStats, spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.verify import (
    CAP_CLASSIC,
    CAP_SHIFTED,
    EchoToken,
    EndpointCapability,
    Prediction,
    ScoreResult,
    generate_burst,
    probe_endpoint,
    score_completion,
)

__all__ = [
    "CAP_CLASSIC",
    "CAP_SHIFTED",
    "EchoToken",
    "EndpointCapability",
    "LookupMemory",
    "Prediction",
    "ScoreResult",
    "SpecStats",
    "generate_burst",
    "probe_endpoint",
    "score_completion",
    "spec_generate",
]
