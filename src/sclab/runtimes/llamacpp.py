from __future__ import annotations

import shutil
import subprocess

from sclab.runtimes.base import ApproxTokenCounterMixin, GenerationRequest, GenerationResult
from sclab.tokenization import count_tokens
from sclab.utils.timing import elapsed_timer


class LlamaCppRuntime(ApproxTokenCounterMixin):
    name = "llama.cpp"

    def __init__(self, executable: str = "llama-cli") -> None:
        self.executable = executable

    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        command = [
            self.executable,
            "-m",
            request.model,
            "-p",
            request.prompt,
            "-n",
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
            "timing_note": "llama.cpp CLI adapter captures wall-clock time; use llama-bench for low-level timing",
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
