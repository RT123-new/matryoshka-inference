from __future__ import annotations

import json
import time
from typing import Any

import requests

from sclab.runtimes.base import ApproxTokenCounterMixin, GenerationRequest, GenerationResult


def _ns_to_s(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000
    except (TypeError, ValueError):
        return None


class OllamaRuntime(ApproxTokenCounterMixin):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.ok
        except requests.RequestException:
            return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        think = request.runtime_options.get("think", False)
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True,
            "think": think,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "seed": request.seed,
            },
        }
        start = time.perf_counter()
        first_token_s: float | None = None
        chunks: list[str] = []
        final_metadata: dict[str, Any] = {}
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=request.timeout_s,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    piece = event.get("response") or ""
                    if piece and first_token_s is None:
                        first_token_s = time.perf_counter() - start
                    chunks.append(piece)
                    if event.get("done"):
                        final_metadata = event
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - start
            return GenerationResult(
                text="",
                model=request.model,
                runtime=self.name,
                prompt_tokens=None,
                completion_tokens=None,
                total_time_s=elapsed,
                time_to_first_token_s=first_token_s,
                prompt_eval_time_s=None,
                decode_time_s=None,
                decode_tokens_per_s=None,
                raw_metadata={"error": str(exc), "payload_model": request.model},
            )

        total_time = time.perf_counter() - start
        prompt_eval_time = _ns_to_s(final_metadata.get("prompt_eval_duration"))
        decode_time = _ns_to_s(final_metadata.get("eval_duration"))
        completion_tokens = final_metadata.get("eval_count")
        decode_tps = None
        if decode_time and completion_tokens:
            decode_tps = float(completion_tokens) / decode_time
        return GenerationResult(
            text="".join(chunks).strip(),
            model=request.model,
            runtime=self.name,
            prompt_tokens=final_metadata.get("prompt_eval_count"),
            completion_tokens=completion_tokens,
            total_time_s=_ns_to_s(final_metadata.get("total_duration")) or total_time,
            time_to_first_token_s=first_token_s,
            prompt_eval_time_s=prompt_eval_time,
            decode_time_s=decode_time,
            decode_tokens_per_s=decode_tps,
            raw_metadata={**final_metadata, "request_think": think},
        )
