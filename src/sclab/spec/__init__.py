"""Universal API-level verified speculation (experimental).

Turns any OpenAI-compatible engine that supports completion *scoring*
(``echo`` + ``logprobs`` on ``/v1/completions``) into a lossless speculative
decoder — through its public API, with no engine patch, no special checkpoint,
and no draft model required. See HANDOFF.md for the full plan.
"""

from sclab.spec.loop import SpecStats, spec_generate
from sclab.spec.memory import LookupMemory
from sclab.spec.verify import EchoToken, ScoreResult, generate_burst, score_completion

__all__ = [
    "EchoToken",
    "LookupMemory",
    "ScoreResult",
    "SpecStats",
    "generate_burst",
    "score_completion",
    "spec_generate",
]
