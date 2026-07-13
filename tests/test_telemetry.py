"""TelemetryStore: per-request sessions, aggregates, and the dashboard snapshot.

The proxy backend serves requests concurrently, so overlapping sessions must
keep independent token counts instead of trampling one shared slot.
"""

from __future__ import annotations

from sclab.runtimes.orthrus_engine import DecodeTelemetry, StepRecord
from sclab.telemetry import TelemetryStore


def test_single_session_records_history_row():
    store = TelemetryStore()
    store.start("r1", "proxy", "hello world")
    store.tick("r1", n=3)
    store.tick("r1")
    store.finish("r1", prompt_tokens=42)

    snap = store.snapshot()
    assert snap["live"]["active"] is False
    assert snap["totals"]["requests"] == 1
    assert snap["totals"]["tokens"] == 4
    assert snap["totals"]["by_mode"]["proxy"] == 1
    assert snap["totals"]["tokens_by_source"]["model"] == 4
    row = snap["history"][0]
    assert row["request_id"] == "r1"
    assert row["tokens"] == 4
    assert row["prompt_tokens"] == 42
    assert row["accepted_per_pass"] is None  # not meaningful for proxy


def test_concurrent_sessions_do_not_trample_each_other():
    store = TelemetryStore()
    store.start("a", "proxy", "first")
    store.start("b", "proxy", "second")
    store.tick("a", n=10)
    store.tick("b", n=1)
    # Finish in start order: a's row must carry a's tokens, not b's.
    store.finish("a")
    store.finish("b")

    rows = {r["request_id"]: r for r in store.snapshot()["history"]}
    assert rows["a"]["tokens"] == 10
    assert rows["b"]["tokens"] == 1
    assert store.snapshot()["totals"]["tokens"] == 11
    assert store.snapshot()["totals"]["requests"] == 2


def test_snapshot_live_follows_most_recent_active_session():
    store = TelemetryStore()
    store.start("a", "proxy", "first")
    store.start("b", "proxy", "second")
    store.tick("b", n=5)
    snap = store.snapshot()
    assert snap["live"]["active"] is True
    assert snap["live"]["prompt_preview"] == "second"
    assert snap["live"]["tokens"] == 5
    assert snap["live"]["concurrent"] == 2
    store.finish("b")
    assert store.snapshot()["live"]["prompt_preview"] == "first"
    store.finish("a")
    assert store.snapshot()["live"]["active"] is False


def test_set_tokens_records_whole_response_at_once():
    store = TelemetryStore()
    store.start("r1", "proxy", "p")
    store.set_tokens("r1", 128)
    store.finish("r1")
    assert store.snapshot()["history"][0]["tokens"] == 128


def test_finish_unknown_request_is_a_noop():
    store = TelemetryStore()
    store.finish("never-started")
    store.tick("never-started")
    snap = store.snapshot()
    assert snap["totals"]["requests"] == 0
    assert snap["history"] == []


def _diffusion_telemetry() -> DecodeTelemetry:
    tel = DecodeTelemetry(mode="diffusion")
    tel.steps.append(StepRecord("diffusion", block_size=8, proposed=7, accepted=5, emitted=6, entropy=0.1))
    tel.steps.append(StepRecord("ar", block_size=1, proposed=0, accepted=0, emitted=1, entropy=0.2))
    tel.tokens_generated = 7
    return tel


def test_orthrus_summary_feeds_aggregates():
    store = TelemetryStore()
    tel = _diffusion_telemetry()
    store.start("r1", "diffusion", "make json")
    for _ in range(tel.tokens_generated):
        store.tick("r1", tel)
    store.finish("r1", tel, prompt_tokens=9)

    snap = store.snapshot()
    assert snap["totals"]["by_mode"]["diffusion"] == 1
    # source mix from the decode telemetry, not the proxy "model" bucket
    assert snap["totals"]["tokens_by_source"]["diffusion"] == 6
    assert snap["totals"]["tokens_by_source"]["ar"] == 1
    assert snap["totals"]["tokens_by_source"]["model"] == 0
    # accepted draft tokens come from the real accepted count (5), not a
    # rate-times-passes approximation
    assert store.accepted_from_draft == 5
    row = snap["history"][0]
    assert row["accepted_per_pass"] == 3.5  # 7 tokens / 2 passes
    assert row["prompt_tokens"] == 9


def test_summary_exposes_draft_token_counts():
    summary = _diffusion_telemetry().summary()
    assert summary["draft_tokens_proposed"] == 7
    assert summary["draft_tokens_accepted"] == 5
