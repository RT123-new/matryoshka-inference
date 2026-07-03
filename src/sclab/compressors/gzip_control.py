from __future__ import annotations

import base64
import gzip

from sclab.compressors.base import Document, build_result


class GzipB64ControlCompressor:
    name = "gzip_b64_control"

    def compress(self, doc: Document, question: str | None = None):
        payload = base64.b64encode(gzip.compress(doc.text.encode("utf-8"))).decode("ascii")
        text = (
            "GZIP+BASE64 CONTROL PAYLOAD\n"
            "The original source text was gzip-compressed and base64-encoded. "
            "This is a control, not semantic compression.\n\n"
            f"{payload}"
        )
        return build_result(
            doc=doc,
            text=text,
            method=self.name,
            metadata={"control": True, "expected_quality": "poor_for_normal_llms"},
        )
