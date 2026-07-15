"""Verification-backend abstraction for token-ID-level verified speculation.

Phase 1 (text-surface) speculation could only ever prove *surface* identity: it
reads decoded token *text* and character offsets from an OpenAI-compatible
``/v1/completions`` endpoint, so two distinct token ids that decode to the same
string are indistinguishable to it, and a decoded string that re-tokenizes to a
different id sequence silently breaks the seam. See ``docs/spec_phase1_results.md``.

This module defines the seam that Phase 2 verifies *token ids* over instead. A
:class:`VerificationBackend` owns tokenizer semantics, target logits, token ids
and engine policy; the loop (``token_verify.py``) owns proposer selection,
accepted-prefix logic, budgets and telemetry. The two responsibilities are kept
apart on purpose: correctness lives in the backend's *raw-argmax* predictions,
and everything the loop does is bookkeeping on top of them.

Nothing here talks to a network or imports a heavy dependency; the concrete
in-process ``llama-cpp-python`` adapter lives in ``llamacpp_backend.py`` and is
import-guarded so this package still imports with no engine installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# The one policy under which token-ID verification is sound.
# --------------------------------------------------------------------------- #

# Verification reads the target's *raw* per-position argmax. For that to equal
# the token generation emits, generation must decode by raw argmax too — plain
# greedy with *every* history-dependent logit processor disabled. Temperature,
# top-p/top-k/min-p, repetition/presence/frequency penalties, mirostat, grammar
# and logit-bias all make the emitted token depend on more than the raw
# next-token distribution, so none of them may be active. A backend records the
# exact policy it enforced in its :class:`VerificationCapability`.
DETERMINISTIC_POLICY: dict[str, float | int] = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
}

# Fields we know how to neutralise. Anything outside this set, or any known field
# set to a non-deterministic value, must make a policy "unknown" and unusable —
# we never verify against a policy we cannot fully account for.
_KNOWN_POLICY_FIELDS = frozenset(DETERMINISTIC_POLICY) | {
    "grammar", "logit_bias", "response_format", "seed",
}


def policy_is_deterministic(policy: dict) -> tuple[bool, str]:
    """Is ``policy`` exactly raw-argmax greedy with no history dependence?

    Returns ``(ok, reason)``. Rejects any field we do not recognise (so a new
    sampling knob cannot slip through unverified) and any known field set to a
    value that would perturb the raw argmax. A grammar or a non-empty logit bias
    is rejected outright: it changes the argmax in a way text/id verification
    against the *unconstrained* logits does not model.
    """
    for key in policy:
        if key not in _KNOWN_POLICY_FIELDS:
            return False, f"unknown generation-policy field {key!r}"
    if float(policy.get("temperature", 0.0)) != 0.0:
        return False, "temperature != 0"
    if int(policy.get("top_k", 1)) not in (0, 1):
        # 0 means "disabled" on llama.cpp (all tokens), 1 means argmax; both are
        # fine because temperature 0 already forces argmax. Any other cap is not.
        return False, "top_k not in {0, 1}"
    for pen in ("repeat_penalty",):
        if float(policy.get(pen, 1.0)) != 1.0:
            return False, f"{pen} != 1"
    for pen in ("presence_penalty", "frequency_penalty"):
        if float(policy.get(pen, 0.0)) != 0.0:
            return False, f"{pen} != 0"
    if int(policy.get("mirostat", 0)) != 0:
        return False, "mirostat enabled"
    grammar = policy.get("grammar")
    if grammar:
        return False, "grammar constraint active"
    bias = policy.get("logit_bias")
    if bias:
        return False, "logit_bias active"
    return True, "raw-argmax greedy"


# --------------------------------------------------------------------------- #
# Capability classification for a token-ID backend.
# --------------------------------------------------------------------------- #

TOKEN_ID_VERIFIED = "verified_token_id"                 # usable
TOKEN_ID_UNAVAILABLE = "token_id_unavailable"           # no in-process logits path
TOKEN_ID_NONDETERMINISTIC = "token_id_nondeterministic_policy"  # policy not greedy


@dataclass
class VerificationCapability:
    """What a token-ID backend can guarantee about its verification lane.

    A backend must return one of these *before* the verify lane may run; an
    unusable capability forces plain generation. ``policy`` records the exact
    deterministic policy the backend enforced, so the guarantee is auditable.
    """
    mode: str
    deterministic: bool = False
    supports_bonus: bool = True
    eos_token_id: int | None = None
    policy: dict = field(default_factory=dict)
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.mode == TOKEN_ID_VERIFIED and self.deterministic


@dataclass
class GenerationResult:
    """Plain greedy generation over an authoritative context of token ids.

    ``token_ids`` excludes any end-of-sequence token (as a plain client's visible
    output would); ``finish_reason`` is ``"stop"`` when generation ended on EOS,
    ``"length"`` when it hit the token budget, ``None`` only on error.
    """
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class DraftVerification:
    """The target's raw-argmax prediction at each draft position, plus the bonus.

    ``predicted_ids[i]`` is ``argmax`` of the target logits given
    ``context_ids + draft_ids[:i]`` — i.e. the id greedy decoding *would* emit at
    draft position ``i``. There is one extra entry, ``predicted_ids[len(draft)]``,
    the **bonus**: the id greedy would emit *after* a fully accepted draft (the
    token real speculative decoding gets for free from the final verify position).

    The loop compares these to the proposed ``draft_ids`` and owns the
    accepted-prefix / correction / bonus decision (:func:`resolve_draft`). The
    backend never decides acceptance; it only reports argmax over real logits.
    """
    predicted_ids: list[int] = field(default_factory=list)
    error: str | None = None


@runtime_checkable
class VerificationBackend(Protocol):
    """A target engine that can verify token-ID drafts over its real logits."""

    def capability(self) -> VerificationCapability:
        """Report whether the verify lane is usable, and under what policy."""
        ...

    def encode_context(self, text: str) -> list[int]:
        """Tokenize ``text`` to ids exactly as the target would (incl. BOS)."""
        ...

    def decode_tokens(self, token_ids: list[int]) -> bytes:
        """Detokenize ids to raw bytes (the authoritative surface for output)."""
        ...

    def generate_plain(self, context_ids: list[int], max_tokens: int) -> GenerationResult:
        """Plain raw-argmax greedy continuation of ``context_ids``."""
        ...

    def verify_draft(self, context_ids: list[int], draft_ids: list[int]) -> DraftVerification:
        """Target argmax at every draft position over ``context_ids + draft_ids``."""
        ...


def logits_argmax(logits) -> int:
    """Deterministic argmax over a logits row, ties broken by lowest token id.

    Matches llama.cpp's greedy tie-break (lowest id wins) so a fake backend and a
    real one agree at exact ties. ``logits`` may be any indexable sequence of
    floats; non-finite entries are treated as ``-inf`` so a NaN can never win.
    """
    best_id = -1
    best_val = -math.inf
    for i, v in enumerate(logits):
        fv = float(v)
        if not math.isfinite(fv):
            fv = -math.inf
        if fv > best_val:
            best_val = fv
            best_id = i
    return best_id
