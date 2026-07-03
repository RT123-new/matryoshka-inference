from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TokenCount:
    value: int
    method: str
    approximate: bool
    model: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=16)
def _load_hf_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception:
        return None


def count_tokens(text: str, model: str | None = None) -> TokenCount:
    """Count tokens consistently, preferring local exact tokenizers when available."""
    if model:
        tokenizer = _load_hf_tokenizer(model)
        if tokenizer is not None:
            try:
                return TokenCount(
                    value=len(tokenizer.encode(text)),
                    method="huggingface_local",
                    approximate=False,
                    model=model,
                )
            except Exception:
                pass

    # Conservative approximation used by many LLM tools for English-ish text.
    return TokenCount(
        value=max(1, (len(text) + 3) // 4),
        method="chars_div_4",
        approximate=True,
        model=model,
    )
