"""Token-ID-level verified speculation loop.

This is the Phase 2 correctness core. Unlike the text-surface loop
(``loop.py``), the authoritative generation state is a list of **token ids**,
never a decoded string, and every emitted token is one the target's *raw argmax*
would have produced at its position — compared as an **id**, not as text. That
closes the surface-vs-id gaps text mode cannot:

* distinct ids that decode to the same string are distinguished (we compare ids);
* a decoded string that re-tokenizes differently can never corrupt state, because
  we never decode-then-re-tokenize generated output to rebuild context — the ids
  are threaded through directly;
* the text proposer is still allowed, but a proposed *string* is only turned into
  a draft when re-tokenizing ``context_text + draft_text`` reproduces the exact
  authoritative context ids as a prefix; otherwise the draft is rejected and the
  loop takes a plain (still exact) burst.

By induction every accepted-draft, correction and bonus id equals greedy raw-argmax
decoding, so the emitted id sequence — and therefore its bytes — is identical to a
single :meth:`VerificationBackend.generate_plain` call. That equality, on **ids**,
is the correctness gate; byte equality follows from it.

Failure handling is fail-safe, never fail-silent: a backend error mid-run does not
truncate a "successful" result — the loop disables speculation, finishes the
remaining budget with plain generation, preserves the error, and marks the run
``degraded_to_plain`` so the output still equals plain generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sclab.spec.backend import (
    DraftVerification,
    GenerationResult,
    VerificationBackend,
    VerificationCapability,
)
from sclab.spec.memory import LookupMemory


@dataclass
class TokenSpecStats:
    """Disaggregated, id-level telemetry for a token-ID speculation run."""

    requests: int = 0
    token_id_verify_rounds: int = 0
    text_surface_verify_rounds: int = 0   # always 0 here; kept for parity/labeling
    burst_rounds: int = 0

    draft_text_proposals: int = 0
    draft_tokenization_rejected: int = 0
    context_prefix_mismatch: int = 0
    draft_ids_proposed: int = 0
    draft_ids_accepted: int = 0
    correction_ids: int = 0
    bonus_ids: int = 0
    burst_ids: int = 0

    tokens_total: int = 0
    seam_fallbacks: int = 0
    verify_rounds_zero_accept: int = 0

    spec_available: bool = True
    degraded_to_plain: bool = False
    finish_reason: str | None = None
    error: str | None = None
    recent_accepts: list[int] = field(default_factory=list)

    @property
    def draft_ids_accepted_per_verify(self) -> float:
        r = self.token_id_verify_rounds
        return (self.draft_ids_accepted / r) if r else 0.0

    @property
    def tokens_per_request(self) -> float:
        return (self.tokens_total / self.requests) if self.requests else 0.0

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "token_id_verify_rounds": self.token_id_verify_rounds,
            "text_surface_verify_rounds": self.text_surface_verify_rounds,
            "burst_rounds": self.burst_rounds,
            "draft_text_proposals": self.draft_text_proposals,
            "draft_tokenization_rejected": self.draft_tokenization_rejected,
            "context_prefix_mismatch": self.context_prefix_mismatch,
            "draft_ids_proposed": self.draft_ids_proposed,
            "draft_ids_accepted": self.draft_ids_accepted,
            "correction_ids": self.correction_ids,
            "bonus_ids": self.bonus_ids,
            "burst_ids": self.burst_ids,
            "tokens_total": self.tokens_total,
            "seam_fallbacks": self.seam_fallbacks,
            "verify_rounds_zero_accept": self.verify_rounds_zero_accept,
            "draft_ids_accepted_per_verify": round(self.draft_ids_accepted_per_verify, 3),
            "tokens_per_request": round(self.tokens_per_request, 3),
            "spec_available": self.spec_available,
            "degraded_to_plain": self.degraded_to_plain,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class DraftResolution:
    """Outcome of applying one verified draft under the current budget."""
    emitted_ids: list[int]
    n_accepted: int
    n_correction: int
    n_bonus: int
    all_accepted: bool
    finish_reason: str | None = None


def resolve_draft(draft_ids: list[int], predicted_ids: list[int], remaining: int,
                  eos_token_id: int | None = None) -> DraftResolution:
    """Accept the longest exact id prefix, then correct or bonus — the loop's job.

    ``predicted_ids[i]`` is the target's raw-argmax id at draft position ``i``;
    ``predicted_ids[len(draft_ids)]`` (if present) is the bonus after a fully
    accepted draft. Emits at most ``remaining`` ids and never emits an EOS token
    (it stops instead, exactly as plain generation would), so the result can be
    appended to the authoritative context without ever exceeding the budget.
    """
    accepted: list[int] = []
    correction: int | None = None
    for i, d in enumerate(draft_ids):
        if i < len(predicted_ids) and predicted_ids[i] == d:
            accepted.append(d)
        else:
            if i < len(predicted_ids):
                correction = predicted_ids[i]
            break
    all_accepted = len(accepted) == len(draft_ids)

    emitted = accepted[: max(0, remaining)]
    n_accepted = len(emitted)
    n_correction = n_bonus = 0
    finish: str | None = None

    if n_accepted < remaining:
        if correction is not None:
            if correction == eos_token_id:
                finish = "stop"           # stop at EOS, but never emit it
            else:
                emitted = emitted + [correction]
                n_correction = 1
        elif all_accepted and len(predicted_ids) > len(draft_ids):
            bonus = predicted_ids[len(draft_ids)]
            if bonus == eos_token_id:
                finish = "stop"
            else:
                emitted = emitted + [bonus]
                n_bonus = 1
    return DraftResolution(emitted, n_accepted, n_correction, n_bonus, all_accepted, finish)


@dataclass
class TokenGeneration:
    """Result of a token-ID speculation run: authoritative ids + their bytes."""
    token_ids: list[int]
    text_bytes: bytes
    stats: TokenSpecStats

    @property
    def text(self) -> str:
        return self.text_bytes.decode("utf-8", errors="replace")


def _decode_str(backend: VerificationBackend, ids: list[int]) -> str:
    if not ids:
        return ""
    return backend.decode_tokens(ids).decode("utf-8", errors="replace")


def _propose_draft_ids(backend: VerificationBackend, context_ids: list[int],
                       memory: LookupMemory, draft_chars: int, min_draft_chars: int,
                       stats: TokenSpecStats) -> list[int] | None:
    """Turn a text proposal into a token-id draft, conservatively.

    Decodes the *authoritative* context ids to text only to drive the text
    proposer, then re-tokenizes ``context_text + draft_text`` and requires the
    result to begin with the exact authoritative context ids. If it does not
    (a canonicalisation / seam failure — e.g. the draft merges with the context
    tail, or the context itself does not re-tokenize to itself), the draft is
    rejected and the caller falls back to a plain burst. We never assume that
    tokenizing ``draft_text`` on its own is equivalent to the seam-aware suffix.
    """
    context_text = _decode_str(backend, context_ids)
    draft_text = memory.propose(context_text, max_chars=draft_chars, min_chars=min_draft_chars)
    if not draft_text:
        return None
    stats.draft_text_proposals += 1
    full_ids = backend.encode_context(context_text + draft_text)
    if full_ids[: len(context_ids)] != context_ids:
        # Re-tokenizing the context (± the draft) did not reproduce the exact
        # authoritative ids: the text surface is not a faithful proxy here.
        stats.context_prefix_mismatch += 1
        stats.draft_tokenization_rejected += 1
        return None
    draft_ids = full_ids[len(context_ids):]
    if not draft_ids:
        stats.draft_tokenization_rejected += 1
        return None
    stats.draft_ids_proposed += len(draft_ids)
    return draft_ids


def _finish_plain(backend: VerificationBackend, context_ids: list[int], remaining: int,
                  stats: TokenSpecStats) -> tuple[list[int], str | None]:
    """Finish the remaining budget with one plain generation call.

    Used both for the safe fallback (unusable capability, backend error, or a
    proposer we cannot trust) and to guarantee the budget is honoured by the
    engine itself rather than by a token estimate. Returns ``(new_ids, finish)``.
    """
    if remaining <= 0:
        return [], stats.finish_reason
    gen: GenerationResult = backend.generate_plain(context_ids, remaining)
    stats.requests += 1
    if gen.error:
        if stats.error is None:
            stats.error = gen.error
        return [], stats.finish_reason
    new_ids = list(gen.token_ids)[:remaining]
    return new_ids, gen.finish_reason


def spec_generate_tokens(
    backend: VerificationBackend,
    prompt: str,
    max_tokens: int = 256,
    memory: LookupMemory | None = None,
    capability: VerificationCapability | None = None,
    draft_chars: int = 64,
    min_draft_chars: int = 8,
    burst_tokens: int = 16,
    backoff_rounds: int = 4,
) -> TokenGeneration:
    """Greedy generation via token-ID verified speculation.

    The returned ``token_ids`` are **id-identical** to
    ``backend.generate_plain(encode(prompt), max_tokens).token_ids`` for a
    deterministic backend, and ``text_bytes`` is their exact detokenization.
    Pass a usable :class:`VerificationCapability` to enable the verify lane; when
    it is absent or unusable, the whole request is plain generation.
    """
    stats = TokenSpecStats()
    memory = memory if memory is not None else LookupMemory()
    cap = capability if capability is not None else backend.capability()
    prompt_ids = backend.encode_context(prompt)

    # No usable verify lane → plain generation for the whole budget. This is the
    # safe default the public API can never bypass.
    if not cap.usable:
        stats.spec_available = False
        new_ids, finish = _finish_plain(backend, prompt_ids, max_tokens, stats)
        stats.tokens_total = len(new_ids)
        stats.finish_reason = finish or (None if stats.error else "length")
        return TokenGeneration(new_ids, backend.decode_tokens(new_ids) if new_ids else b"", stats)

    memory.observe(prompt)
    context_ids = list(prompt_ids)         # authoritative state — always token ids
    emitted: list[int] = []
    forced_bursts = 0

    while stats.tokens_total < max_tokens:
        remaining = max_tokens - stats.tokens_total
        draft_ids = None
        if forced_bursts == 0:
            draft_ids = _propose_draft_ids(
                backend, context_ids, memory, draft_chars, min_draft_chars, stats)

        # ---- burst: no trustworthy draft, take exact plain tokens ---------- #
        if not draft_ids:
            n = min(burst_tokens, remaining)
            gen = backend.generate_plain(context_ids, n)
            stats.requests += 1
            if gen.error:
                # Fail-safe: finish the rest plainly rather than truncate.
                stats.error = gen.error
                tail, finish = _finish_plain(backend, context_ids, remaining, stats)
                _apply(backend, tail, context_ids, emitted, memory, stats)
                stats.degraded_to_plain = True
                stats.finish_reason = finish or stats.finish_reason
                break
            stats.burst_rounds += 1
            forced_bursts = max(0, forced_bursts - 1)
            new_ids = list(gen.token_ids)[:remaining]
            if not new_ids:
                stats.finish_reason = gen.finish_reason or "stop"
                break
            stats.burst_ids += len(new_ids)
            _apply(backend, new_ids, context_ids, emitted, memory, stats)
            if gen.finish_reason == "stop" and len(gen.token_ids) <= remaining:
                stats.finish_reason = "stop"
                break
            continue

        # ---- verify: one round-trip over context + draft ------------------- #
        stats.token_id_verify_rounds += 1
        dv: DraftVerification = backend.verify_draft(context_ids, draft_ids)
        stats.requests += 1
        if dv.error:
            stats.error = dv.error
            tail, finish = _finish_plain(backend, context_ids, remaining, stats)
            _apply(backend, tail, context_ids, emitted, memory, stats)
            stats.degraded_to_plain = True
            stats.finish_reason = finish or stats.finish_reason
            break

        res = resolve_draft(draft_ids, dv.predicted_ids, remaining, cap.eos_token_id)
        stats.draft_ids_accepted += res.n_accepted
        stats.correction_ids += res.n_correction
        stats.bonus_ids += res.n_bonus
        if res.n_accepted == 0:
            stats.verify_rounds_zero_accept += 1
        stats.recent_accepts.append(res.n_accepted)
        del stats.recent_accepts[:-8]

        _apply(backend, res.emitted_ids, context_ids, emitted, memory, stats)

        if not res.emitted_ids and res.finish_reason != "stop":
            # Nothing emitted and not a stop: force a plain burst so we always
            # make forward progress instead of re-proposing the same wall.
            forced_bursts = 1
        elif len(stats.recent_accepts) >= 3 and sum(stats.recent_accepts[-3:]) == 0:
            forced_bursts = backoff_rounds

        if res.finish_reason == "stop":
            stats.finish_reason = "stop"
            break

    if stats.finish_reason is None and stats.tokens_total >= max_tokens:
        stats.finish_reason = "length"
    return TokenGeneration(list(emitted), backend.decode_tokens(emitted) if emitted else b"", stats)


def _apply(backend: VerificationBackend, new_ids: list[int], context_ids: list[int],
           emitted: list[int], memory: LookupMemory, stats: TokenSpecStats) -> None:
    """Commit ``new_ids`` to the authoritative id state and the text memory.

    The context is extended with the *ids* directly — we never decode and
    re-tokenize generated output to rebuild it. Text is only decoded to feed the
    (text-based) proposer, and only ever as an addition, never as ground truth.
    """
    if not new_ids:
        return
    context_ids.extend(new_ids)
    emitted.extend(new_ids)
    stats.tokens_total += len(new_ids)
    memory.observe(_decode_str(backend, new_ids))
