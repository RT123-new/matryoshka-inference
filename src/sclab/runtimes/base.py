from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from sclab.tokenization import count_tokens


@dataclass
class GenerationRequest:
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    timeout_s: int = 300
    runtime_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    model: str
    runtime: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_time_s: float
    time_to_first_token_s: float | None
    prompt_eval_time_s: float | None
    decode_time_s: float | None
    decode_tokens_per_s: float | None
    raw_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMRuntime(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def count_tokens(self, text: str, model: str | None = None) -> int:
        ...


class ApproxTokenCounterMixin:
    def count_tokens(self, text: str, model: str | None = None) -> int:
        return count_tokens(text, model=model).value
