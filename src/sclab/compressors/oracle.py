from __future__ import annotations

from sclab.compressors.base import Document, build_result


class OracleCompressor:
    name = "oracle"

    def compress(self, doc: Document, question: str | None = None):
        source_span = doc.metadata.get("source_span")
        text = source_span if source_span else doc.text
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={
                "oracle": True,
                "leaderboard_allowed": False,
                "description": "Uses known answer spans for upper-bound testing only",
            },
        )
