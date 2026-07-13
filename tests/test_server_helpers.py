"""Unit tests for the server's request-shaping helpers (no HTTP, no MLX)."""

from __future__ import annotations

from sclab.server import _content_to_text, _is_usage_only_chunk, _user_text


def test_content_to_text_plain_string():
    assert _content_to_text("hello") == "hello"


def test_content_to_text_openai_content_parts():
    parts = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "text", "text": "second"},
    ]
    assert _content_to_text(parts) == "first second"


def test_content_to_text_tolerates_odd_shapes():
    assert _content_to_text(None) == ""
    assert _content_to_text(123) == ""
    assert _content_to_text(["plain", {"no_text": True}]) == "plain"


def test_user_text_joins_only_user_turns():
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "two"}]},
    ]
    assert _user_text(messages) == "one two"


def test_usage_only_chunk_detection():
    usage_only = 'data: {"id":"x","choices":[],"usage":{"completion_tokens":2}}'
    final_with_choices = ('data: {"id":"x","choices":[{"index":0,"delta":{},'
                          '"finish_reason":"stop"}],"usage":{"completion_tokens":2}}')
    content = 'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}'
    assert _is_usage_only_chunk(usage_only) is True
    assert _is_usage_only_chunk(final_with_choices) is False  # carries finish_reason
    assert _is_usage_only_chunk(content) is False
    assert _is_usage_only_chunk("data: [DONE]") is False
    assert _is_usage_only_chunk("") is False
    assert _is_usage_only_chunk("data: not-json") is False
