from __future__ import annotations

from sclab.runtimes.fake import FakeRuntime
from sclab.runtimes.llamacpp import LlamaCppRuntime
from sclab.runtimes.mlx import MLXRuntime
from sclab.runtimes.ollama import OllamaRuntime


def get_runtime(name: str):
    normalized = name.lower()
    if normalized == "fake":
        return FakeRuntime()
    if normalized == "ollama":
        return OllamaRuntime()
    if normalized in {"mlx", "mlx-lm", "mlx_lm"}:
        return MLXRuntime()
    if normalized in {"llama.cpp", "llamacpp", "llama-cpp"}:
        return LlamaCppRuntime()
    if normalized in {"orthrus", "orthrus-mlx", "orthrus_mlx"}:
        # Imported lazily so the base harness works without mlx/transformers.
        from sclab.runtimes.orthrus_mlx import OrthrusMLXRuntime

        return OrthrusMLXRuntime()
    raise ValueError(f"Unknown runtime: {name}")


def available_runtime_names() -> list[str]:
    return ["fake", "ollama", "mlx-lm", "llama.cpp", "orthrus-mlx"]
