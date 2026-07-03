from __future__ import annotations

from sclab.compressors.base import Document, build_result


class RawCompressor:
    name = "raw"

    def compress(self, doc: Document, question: str | None = None):
        return build_result(
            doc=doc,
            text=doc.text,
            method=self.name,
            metadata={"control": True, "description": "Uncompressed source text"},
        )
