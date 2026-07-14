"""Draft-free proposer: a lookup memory over text the session has already seen.

Agent and RAG workloads are massively self-repetitive — tool schemas, quoted
context, code being edited, JSON templates. This memory indexes every piece of
text that flows through the accelerator (prompts and generated output alike)
and, given the current generation context, proposes the continuation that
followed the same suffix last time it appeared. Zero model, zero extra forward
passes; the verifier makes it lossless no matter how wrong a proposal is.

This is the CopySpec / prompt-lookup-decoding idea lifted to plain characters
so it works across *any* tokenizer, and across requests (the memory persists,
so the accelerator gets faster the more you use it).
"""

from __future__ import annotations


class LookupMemory:
    def __init__(self, shingle: int = 16, max_bytes: int = 1_000_000,
                 positions_per_key: int = 8) -> None:
        if shingle < 4:
            raise ValueError("shingle must be >= 4 characters")
        self.shingle = shingle
        self.max_bytes = max_bytes
        self.positions_per_key = positions_per_key
        self._buf = ""
        # shingle -> recent start positions of that shingle in _buf (oldest first)
        self._index: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._buf)

    def observe(self, text: str) -> None:
        """Feed text (a prompt, or freshly generated output) into the memory."""
        if not text:
            return
        start = max(0, len(self._buf) - self.shingle + 1)
        self._buf += text
        for i in range(start, len(self._buf) - self.shingle + 1):
            self._add(self._buf[i : i + self.shingle], i)
        if len(self._buf) > self.max_bytes:
            self._trim()

    def propose(self, context: str, max_chars: int = 64, min_chars: int = 8) -> str | None:
        """Propose a continuation of ``context`` seen after the same suffix before.

        Returns up to ``max_chars`` of the text that followed the most recent
        *earlier* occurrence of the context's last ``shingle`` characters, or
        ``None`` when there is no useful match. The self-occurrence at the very
        end of the buffer is skipped (it has nothing after it), and the
        proposal is cut back to the last whitespace so it usually ends on a
        clean token boundary — which keeps the verifier's seam check happy.
        """
        if len(context) < self.shingle:
            return None
        tail = context[-self.shingle :]
        positions = self._index.get(tail)
        if not positions:
            return None
        # Exclude the trailing self-match: context is a suffix of _buf, so its
        # tail sits at buffer position len(_buf) - shingle.
        limit = len(self._buf) - self.shingle
        for pos in reversed(positions):
            if pos >= limit:
                continue
            candidate = self._buf[pos + self.shingle : pos + self.shingle + max_chars]
            if len(candidate) < min_chars:
                continue
            if len(candidate) == max_chars:
                cut = candidate.rfind(" ")
                if cut >= min_chars:
                    candidate = candidate[: cut + 1]
            if len(candidate) >= min_chars:
                return candidate
        return None

    def _add(self, key: str, pos: int) -> None:
        bucket = self._index.get(key)
        if bucket is None:
            self._index[key] = [pos]
        else:
            bucket.append(pos)
            if len(bucket) > self.positions_per_key:
                del bucket[0]

    def _trim(self) -> None:
        """Drop the oldest half of the buffer and rebuild the index."""
        keep_from = len(self._buf) // 2
        self._buf = self._buf[keep_from:]
        self._index = {}
        for i in range(0, len(self._buf) - self.shingle + 1):
            self._add(self._buf[i : i + self.shingle], i)
