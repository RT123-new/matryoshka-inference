from __future__ import annotations

from sclab.compressors.base import Compressor, CompressionResult, Document
from sclab.compressors.extractive import ExtractiveRelevanceCompressor
from sclab.compressors.fact_table import FactTableCompressor
from sclab.compressors.gzip_control import GzipB64ControlCompressor
from sclab.compressors.hybrid import HybridBriefExcerptsCompressor
from sclab.compressors.oracle import OracleCompressor
from sclab.compressors.raw import RawCompressor
from sclab.compressors.semantic_brief import SemanticBriefCompressor


def get_compressor(name: str, budget: float | None = None) -> Compressor:
    normalized = name.strip()
    if normalized == "raw":
        return RawCompressor()
    if normalized == "gzip_b64_control":
        return GzipB64ControlCompressor()
    if normalized == "extractive_relevance":
        return ExtractiveRelevanceCompressor(target_token_ratio=budget or 0.35)
    if normalized == "semantic_brief":
        return SemanticBriefCompressor(target_token_ratio=budget)
    if normalized == "fact_table":
        return FactTableCompressor()
    if normalized == "hybrid_brief_excerpts":
        return HybridBriefExcerptsCompressor(target_token_ratio=budget or 0.30)
    if normalized == "oracle":
        return OracleCompressor()
    raise ValueError(f"Unknown compressor: {name}")


def compressor_names() -> list[str]:
    return [
        "raw",
        "gzip_b64_control",
        "extractive_relevance",
        "semantic_brief",
        "fact_table",
        "hybrid_brief_excerpts",
        "oracle",
    ]


__all__ = [
    "Compressor",
    "CompressionResult",
    "Document",
    "get_compressor",
    "compressor_names",
]
