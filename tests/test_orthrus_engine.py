"""Unit tests for the model-free parts of the Orthrus decode engine:
block-sizing policy, copy proposer, and telemetry math. These do not require
mlx or a downloaded checkpoint."""

from __future__ import annotations

from sclab.runtimes.orthrus_engine import (
    BlockPolicy,
    CopyProposer,
    DecodeTelemetry,
    StepRecord,
    _think_state,
    prune_draft,
    route_mode,
)


def test_copy_proposer_matches_repeated_span():
    cp = CopyProposer(ngram=4, min_match=6)
    # A repeated template: the second time we see the prefix, we should be able
    # to copy what followed it the first time.
    first = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    cp.extend(first)
    cp.extend([99])  # some divergence
    cp.extend([10, 11, 12, 13, 14, 15, 16])  # re-enter the template suffix
    proposal = cp.propose(want=3)
    assert proposal == [17, 18, 19]


def test_copy_proposer_returns_none_without_repeat():
    cp = CopyProposer(ngram=4, min_match=6)
    cp.extend([1, 2, 3, 4, 5, 6, 7, 8])
    assert cp.propose(want=4) is None


def test_block_policy_fixed_is_constant():
    p = BlockPolicy(mode="fixed", block_size=8)
    assert p.next_block_size() == 8
    p.update(accepted=0, proposed=7)
    assert p.next_block_size() == 8


def test_block_policy_adaptive_grows_on_high_acceptance():
    p = BlockPolicy(mode="adaptive", block_size=6, min_block=2, max_block=16, window=4)
    for _ in range(4):
        p.update(accepted=7, proposed=7)  # 100% acceptance
    grown = p.next_block_size()
    assert grown > 6


def test_block_policy_adaptive_shrinks_on_low_acceptance():
    p = BlockPolicy(mode="adaptive", block_size=8, min_block=2, max_block=16, window=4)
    for _ in range(4):
        p.update(accepted=0, proposed=7)  # 0% acceptance
    assert p.next_block_size() < 8


def test_block_policy_entropy_gate_caps_block():
    p = BlockPolicy(mode="adaptive", block_size=12, min_block=2, max_block=16, window=4)
    for _ in range(4):
        p.update(accepted=7, proposed=7)
    capped = p.next_block_size(last_entropy=5.0)  # high uncertainty
    assert capped <= 4


def test_block_policy_structured_text_forces_max():
    p = BlockPolicy(mode="adaptive", block_size=4, min_block=2, max_block=16, window=4)
    bs = p.next_block_size(recent_text='{"customer_id": "abc",')
    assert bs == 16


def test_block_policy_adaptive_stays_at_sweet_spot_not_max():
    # High acceptance should grow toward the sweet spot (16), not max_block (32).
    p = BlockPolicy(mode="adaptive", block_size=8, min_block=2, max_block=32,
                    window=4, structured_block=16)
    for _ in range(20):
        p.update(accepted=7, proposed=7)
        p.next_block_size()
    assert p.next_block_size() <= 16


def test_router_sends_structured_to_diffusion():
    assert route_mode("Output a JSON array of user objects.")[0] == "diffusion"
    assert route_mode("Write a Python function to sort a list.")[0] == "diffusion"
    assert route_mode("Solve step by step: 2x+3=11.")[0] == "diffusion"


def test_router_sends_prose_to_ar():
    assert route_mode("Explain why the sky is blue in a paragraph.")[0] == "ar"
    assert route_mode("Describe your favorite holiday.")[0] == "ar"
    assert route_mode("Tell me something interesting.")[0] == "ar"  # no cue -> ar


def test_prune_draft_keeps_confident_prefix():
    # cum: 0.9, 0.72, 0.288 -> with tau=0.3 keep 2
    assert prune_draft([0.9, 0.8, 0.4], tau=0.3) == 2


def test_prune_draft_all_confident_keeps_everything():
    assert prune_draft([0.99, 0.98, 0.97], tau=0.5) == 3


def test_prune_draft_unconfident_first_token_prunes_to_zero():
    assert prune_draft([0.1, 0.9, 0.9], tau=0.3) == 0


def test_scheduled_policy_backs_off_to_ar_lane_and_reprobes():
    p = BlockPolicy(mode="scheduled", block_size=8, min_block=2, max_block=16,
                    window=4, backoff_steps=3, probe_block=6)
    for _ in range(3):
        p.update(accepted=0, proposed=7)  # collapse -> enter backoff
    lane = [p.next_block_size() for _ in range(4)]
    assert lane[:3] == [1, 1, 1]          # AR lane for backoff_steps
    assert lane[3] == 6                    # then re-probe with probe_block


def test_scheduled_policy_stays_speculative_on_good_acceptance():
    p = BlockPolicy(mode="scheduled", block_size=8, min_block=2, max_block=16,
                    window=4, backoff_steps=3)
    for _ in range(4):
        p.update(accepted=7, proposed=7)
    assert p.next_block_size() > 1


def test_think_state_tracking():
    assert _think_state(False, "let me <think> about") is True
    assert _think_state(True, "done </think> answer:") is False
    assert _think_state(True, "no tags here") is True
    assert _think_state(False, "<think>x</think>") is False


def test_policy_think_lane_boosts_block():
    p = BlockPolicy(mode="scheduled", block_size=4, min_block=2, max_block=16,
                    structured_block=16)
    assert p.next_block_size(in_think=True) == 16


def test_telemetry_accepted_per_pass():
    tel = DecodeTelemetry(mode="diffusion")
    tel.steps.append(StepRecord("diffusion", block_size=8, proposed=7, accepted=5, emitted=6, entropy=0.1))
    tel.steps.append(StepRecord("diffusion", block_size=8, proposed=7, accepted=3, emitted=4, entropy=0.2))
    tel.tokens_generated = 10
    assert tel.verification_passes == 2
    assert tel.accepted_tokens_per_verification_pass == 5.0
    assert abs(tel.draft_acceptance_rate - (8 / 14)) < 1e-9
