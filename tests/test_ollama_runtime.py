from sclab.runtimes.base import GenerationRequest
from sclab.runtimes.ollama import OllamaRuntime


def test_ollama_runtime_disables_thinking_by_default(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield '{"response":"ok","done":false}'
            yield '{"response":"","done":true,"prompt_eval_count":4,"eval_count":1,"eval_duration":100000000,"total_duration":200000000}'

    def fake_post(url, json, timeout, stream):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("sclab.runtimes.ollama.requests.post", fake_post)
    result = OllamaRuntime().generate(GenerationRequest(model="m", prompt="p"))

    assert captured["payload"]["think"] is False
    assert result.raw_metadata["request_think"] is False
    assert result.text == "ok"


class _StreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        yield from self._lines


def test_ollama_runtime_surfaces_mid_stream_error(monkeypatch):
    # Ollama reports failures (model OOM, ...) as an error event on a 200
    # response; the harness must record it instead of returning a silent
    # empty answer that scores as a compression failure.
    lines = ['{"response":"par","done":false}', '{"error":"model ran out of memory"}']
    monkeypatch.setattr("sclab.runtimes.ollama.requests.post",
                        lambda url, json, timeout, stream: _StreamResponse(lines))
    result = OllamaRuntime().generate(GenerationRequest(model="m", prompt="p"))

    assert result.text == ""
    assert result.raw_metadata["error"] == "model ran out of memory"
    assert result.completion_tokens is None


def test_ollama_runtime_skips_malformed_stream_lines(monkeypatch):
    lines = [
        '{"response":"ok","done":false}',
        "not json at all",
        '{"response":"!","done":true,"eval_count":2,"eval_duration":100000000}',
    ]
    monkeypatch.setattr("sclab.runtimes.ollama.requests.post",
                        lambda url, json, timeout, stream: _StreamResponse(lines))
    result = OllamaRuntime().generate(GenerationRequest(model="m", prompt="p"))

    assert result.text == "ok!"
    assert result.completion_tokens == 2
