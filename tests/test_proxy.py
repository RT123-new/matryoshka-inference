"""Unit tests for the model-agnostic proxy helpers."""

from __future__ import annotations

from sclab.proxy import count_delta_tokens


def test_count_delta_tokens_content():
    assert count_delta_tokens(
        {"choices": [{"delta": {"content": "hi"}}]}) == 1


def test_count_delta_tokens_tool_call_arguments():
    chunk = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "get_weather", "arguments": '{"city"'}},
    ]}}]}
    assert count_delta_tokens(chunk) == 1


def test_count_delta_tokens_empty_shapes():
    assert count_delta_tokens({}) == 0
    assert count_delta_tokens({"choices": []}) == 0
    assert count_delta_tokens({"choices": [{"delta": {}}]}) == 0
    assert count_delta_tokens({"choices": [{"delta": {"content": ""}}]}) == 0
