from __future__ import annotations

from sclab.compressors.base import Document, build_result
from sclab.compressors.extractive import ExtractiveRelevanceCompressor
from sclab.compressors.semantic_brief import SemanticBriefCompressor


class HybridBriefExcerptsCompressor:
    name = "hybrid_brief_excerpts"

    def __init__(self, target_token_ratio: float = 0.30) -> None:
        self.target_token_ratio = target_token_ratio
        self.brief = SemanticBriefCompressor(max_items=5)
        self.extractive = ExtractiveRelevanceCompressor(target_token_ratio=target_token_ratio, max_paragraphs=4)

    def compress(self, doc: Document, question: str | None = None):
        brief = self.brief.compress(doc, question)
        excerpts = self.extractive.compress(doc, question)
        text = (
            "HYBRID COMPRESSED SOURCE\n\n"
            "Warning: this is a compressed representation. Missing evidence should be reported as missing.\n\n"
            f"{brief.compressed_text}\n\n"
            "SELECTED EXACT EXCERPTS\n"
            f"{excerpts.compressed_text}"
        )
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={
                "target_token_ratio": self.target_token_ratio,
                "components": {
                    "brief": brief.to_dict(),
                    "extractive": excerpts.to_dict(),
                },
            },
        )
