from sclab.runtimes.base import GenerationRequest
from sclab.runtimes.fake import FakeRuntime


def test_fake_runtime_returns_generation_contract():
    runtime = FakeRuntime()
    result = runtime.generate(
        GenerationRequest(
            model="fake",
            prompt="Source:\nMonthly rent is GBP 1,100 per month.\n\nQuestion:\nWhat is the monthly rent?\n\nAnswer:\n",
            runtime_options={
                "task": {
                    "source_span": "Monthly rent is GBP 1,100 per month",
                    "gold_answer": "GBP 1,100 per month",
                    "must_include": ["1,100"],
                    "type": "single_fact",
                }
            },
        )
    )
    assert result.runtime == "fake"
    assert result.prompt_tokens is not None
    assert "1,100" in result.text
