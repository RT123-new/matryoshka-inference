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
