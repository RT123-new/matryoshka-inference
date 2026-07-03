from __future__ import annotations

import shutil
import subprocess

from sclab.runtimes.base import ApproxTokenCounterMixin, GenerationRequest, GenerationResult
from sclab.tokenization import count_tokens
from sclab.utils.timing import elapsed_timer


class MLXRuntime(ApproxTokenCounterMixin):
    name = "mlx-lm"

    def __init__(self, executable: str = "mlx_lm.generate") -> None:
        self.executable = executable

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        command = [
            self.executable,
            "--model",
            request.model,
            "--prompt",
            request.prompt,
            "--max-tokens",
            str(request.max_tokens),
            "--temp",
            str(request.temperature),
        ]
        with elapsed_timer() as timer:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=request.timeout_s,
                check=False,
            )
        text = completed.stdout.strip()
        metadata = {
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "timing_note": "mlx-lm CLI timing is wall-clock only in this adapter",
        }
        return GenerationResult(
            text=text,
            model=request.model,
            runtime=self.name,
            prompt_tokens=count_tokens(request.prompt, model=request.model).value,
            completion_tokens=count_tokens(text, model=request.model).value if text else 0,
            total_time_s=timer.elapsed_s,
            time_to_first_token_s=None,
            prompt_eval_time_s=None,
            decode_time_s=None,
            decode_tokens_per_s=None,
            raw_metadata=metadata,
        )
