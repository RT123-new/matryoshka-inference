from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from sclab.tokenization import count_tokens


@dataclass
class Document:
    text: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    compressed_text: str
    method: str
    original_chars: int
    compressed_chars: int
    original_tokens: int | None
    compressed_tokens: int | None
    compression_ratio_tokens: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Compressor(Protocol):
    name: str

    def compress(self, doc: Document, question: str | None = None) -> CompressionResult:
        ...


def build_result(
    *,
    doc: Document,
    text: str,
    method: str,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CompressionResult:
    original_count = count_tokens(doc.text, model=model)
    compressed_count = count_tokens(text, model=model)
    ratio = None
    if original_count.value:
        ratio = compressed_count.value / original_count.value
    merged_metadata = {
        "token_count_original": original_count.to_dict(),
        "token_count_compressed": compressed_count.to_dict(),
    }
    if metadata:
        merged_metadata.update(metadata)
    return CompressionResult(
        compressed_text=text,
        method=method,
        original_chars=len(doc.text),
        compressed_chars=len(text),
        original_tokens=original_count.value,
        compressed_tokens=compressed_count.value,
        compression_ratio_tokens=ratio,
        metadata=merged_metadata,
    )
