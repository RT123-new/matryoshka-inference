from __future__ import annotations

import re
import time

from sclab.runtimes.base import ApproxTokenCounterMixin, GenerationRequest, GenerationResult
from sclab.tokenization import count_tokens
from sclab.utils.numbers_dates import extract_dates, extract_numbers


class FakeRuntime(ApproxTokenCounterMixin):
    name = "fake"

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        task = request.runtime_options.get("task") or {}
        context = _extract_context(request.prompt)
        text = ""
        source_span = task.get("source_span") or ""
        gold = task.get("gold_answer") or ""
        must_include = task.get("must_include") or []

        if source_span and _appears_in_context(source_span, context):
            text = gold or source_span
        elif gold and _appears_in_context(gold, context):
            text = gold
        elif must_include and all(str(item).lower() in context.lower() for item in must_include):
            text = gold or ", ".join(str(item) for item in must_include)
        else:
            numbers = extract_numbers(context)
            dates = extract_dates(context)
            if task.get("type") in {"number_date", "single_fact"} and (numbers or dates):
                text = " ".join((dates + numbers)[:3])
            else:
                text = "Not found in compressed source." if "compressed" in request.prompt[:500].lower() else "Not found in source."

        # Simulate prompt processing cost so compression can show directionally meaningful timings in CI.
        prompt_tokens = count_tokens(request.prompt, model=request.model).value
        completion_tokens = count_tokens(text, model=request.model).value
        prompt_eval = prompt_tokens * 0.00002
        decode_time = completion_tokens * 0.00005
        total = max(time.perf_counter() - start, prompt_eval + decode_time)
        return GenerationResult(
            text=text,
            model=request.model,
            runtime=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_time_s=total,
            time_to_first_token_s=prompt_eval,
            prompt_eval_time_s=prompt_eval,
            decode_time_s=decode_time,
            decode_tokens_per_s=(completion_tokens / decode_time) if decode_time else None,
            raw_metadata={"fake_runtime": True, "not_for_real_conclusions": True},
        )


def _extract_context(prompt: str) -> str:
    patterns = [
        r"Source:\n(?P<context>.*?)\n\nQuestion:",
        r"Compressed source:\n(?P<context>.*?)\n\nQuestion:",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.DOTALL)
        if match:
            return match.group("context")
    return prompt


def _appears_in_context(needle: str, context: str) -> bool:
    if not needle:
        return False
    normalized_context = re.sub(r"\s+", " ", context).lower()
    normalized_needle = re.sub(r"\s+", " ", needle).lower()
    if normalized_needle in normalized_context:
        return True
    pieces = [part.strip().lower() for part in re.split(r"[.;,\n]", needle) if len(part.strip()) >= 4]
    return bool(pieces) and any(piece in normalized_context for piece in pieces)
