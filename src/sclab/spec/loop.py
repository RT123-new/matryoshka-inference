"""The lossless speculation loop: draft from memory, verify by scoring.

Every emitted token is one of:

* a **burst** token — plain greedy generation from the engine (what a
  non-accelerated client would have received),
* an **accepted draft** token — verified *exactly* equal to the engine's
  unambiguous greedy choice at its position by a scoring round-trip,
* a **correction** token — the engine's greedy choice at the first position
  where the draft diverged (only when that choice is itself unambiguous),
* a **bonus** token — the single token the engine generates past a fully
  accepted draft.

By induction all four are exactly what greedy decoding would have produced *for
a raw-argmax policy* (see ``verify._GREEDY_POLICY``), so the final text is
byte-identical to running the engine plainly with that policy — the speedup
comes purely from verifying many tokens per round-trip instead of one per
sequential decode step.

Two safety rules, both learned the hard way against real engines:

* **Positional shift.** ``score_completion`` must be told the endpoint's
  measured ``shift`` (see ``verify.probe_endpoint``); the wrong shift silently
  drops or duplicates tokens. Callers that speculate against an unprobed or
  unusable endpoint must fall back to plain generation instead.
* **Never stall, never guess.** If a draft token is not the unambiguous greedy
  choice and the engine offers no usable correction (a tie, an empty/special
  surface, or a tokenization seam), the round emits nothing speculative and the
  loop takes a plain burst to step past — it never re-proposes the same
  unverifiable token in a loop, and never emits a token it could not verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sclab.spec.memory import LookupMemory
from sclab.spec.verify import EndpointCapability, generate_burst, score_completion


@dataclass
class SpecStats:
    requests: int = 0
    verify_rounds: int = 0
    burst_rounds: int = 0
    tokens_total: int = 0
    tokens_accepted: int = 0     # REAL draft tokens verified equal to greedy
    tokens_correction: int = 0
    tokens_bonus: int = 0
    tokens_burst: int = 0
    seam_fallbacks: int = 0
    verify_rounds_zero_accept: int = 0   # verify rounds that accepted 0 draft tokens
    spec_available: bool = True          # False when the capability was unusable
    degraded_to_plain: bool = False      # a mid-run failure forced plain generation
    usage_fallback: bool = False         # missing token counts forced one plain finish
    finish_reason: str | None = None
    error: str | None = None
    recent_accepts: list[int] = field(default_factory=list)

    # -- honest, disaggregated per-verify telemetry ----------------------- #
    @property
    def draft_tokens_accepted_per_verify(self) -> float:
        """Accepted *draft* tokens only — the actual speculation win."""
        return (self.tokens_accepted / self.verify_rounds) if self.verify_rounds else 0.0

    @property
    def tokens_emitted_per_verify(self) -> float:
        """All tokens a verify round yields: accepted draft + correction + bonus."""
        if not self.verify_rounds:
            return 0.0
        return (self.tokens_accepted + self.tokens_correction + self.tokens_bonus) / self.verify_rounds

    @property
    def corrections_per_verify(self) -> float:
        return (self.tokens_correction / self.verify_rounds) if self.verify_rounds else 0.0

    @property
    def bonus_tokens_per_verify(self) -> float:
        return (self.tokens_bonus / self.verify_rounds) if self.verify_rounds else 0.0

    @property
    def accepted_per_verify(self) -> float:
        """Backward-compatible alias: accepted *draft* tokens per verify round.

        Historically this also folded in correction and bonus tokens, which
        made a run that accepted **zero** drafts but emitted one correction per
        round look like "1.0 accepted/verify". It now means what its name says;
        use :attr:`tokens_emitted_per_verify` for the old, broader quantity.
        """
        return self.draft_tokens_accepted_per_verify

    @property
    def tokens_per_request(self) -> float:
        """Emitted tokens per engine round-trip. Plain decoding is bounded at 1
        token per decode step; above ~1 per request here is time bought back."""
        return (self.tokens_total / self.requests) if self.requests else 0.0

    def summary(self) -> dict:
        return {
            "requests": self.requests,
            "verify_rounds": self.verify_rounds,
            "burst_rounds": self.burst_rounds,
            "tokens_total": self.tokens_total,
            "tokens_accepted": self.tokens_accepted,
            "tokens_correction": self.tokens_correction,
            "tokens_bonus": self.tokens_bonus,
            "tokens_burst": self.tokens_burst,
            "seam_fallbacks": self.seam_fallbacks,
            "verify_rounds_zero_accept": self.verify_rounds_zero_accept,
            "spec_available": self.spec_available,
            "degraded_to_plain": self.degraded_to_plain,
            "usage_fallback": self.usage_fallback,
            "draft_tokens_accepted_per_verify": round(self.draft_tokens_accepted_per_verify, 3),
            "tokens_emitted_per_verify": round(self.tokens_emitted_per_verify, 3),
            "corrections_per_verify": round(self.corrections_per_verify, 3),
            "bonus_tokens_per_verify": round(self.bonus_tokens_per_verify, 3),
            "accepted_per_verify": round(self.accepted_per_verify, 3),
            "tokens_per_request": round(self.tokens_per_request, 3),
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


def spec_generate(
    upstream: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    memory: LookupMemory | None = None,
    capability: EndpointCapability | None = None,
    draft_chars: int = 64,
    min_draft_chars: int = 8,
    burst_tokens: int = 16,
    backoff_rounds: int = 4,
    timeout: int = 600,
) -> tuple[str, SpecStats]:
    """Generate greedily via the engine's public API, faster where possible.

    Returns ``(generated_text, stats)``. The text is byte-identical to what a
    single plain greedy request (with ``verify._GREEDY_POLICY``) would produce.

    Speculation runs **only** when ``capability`` is a *usable* endpoint
    capability (see ``verify.probe_endpoint``); without one, the request is
    plain generation. This is the safety gate: there is no public default that
    assumes an alignment. Even usable, text mode proves *surface* identity only,
    not token-id identity — use ``token_verify.spec_generate_tokens`` with a
    token-ID backend for unconditional equivalence.

    Token budgeting never guesses: burst tokens are counted from the engine's
    ``completion_tokens`` usage. If that is absent, incremental speculation is
    disabled and the remaining budget is finished in one plain call, so a server
    with no usage data can never drive output past ``max_tokens``.
    """
    memory = memory if memory is not None else LookupMemory()
    stats = SpecStats()

    # No usable capability → plain generation, full stop. The engine's own
    # ``max_tokens`` bounds the output; nothing here can silently mis-verify.
    if capability is None or not capability.usable:
        stats.spec_available = False
        return _plain_only(upstream, api_key, model, prompt, max_tokens, stats, timeout)
    shift = capability.shift or 0

    memory.observe(prompt)
    ctx = prompt
    out = ""
    forced_bursts = 0   # anti-thrash: after bad verify rounds, burst for a while

    while stats.tokens_total < max_tokens:
        remaining = max_tokens - stats.tokens_total
        draft = None
        if forced_bursts == 0:
            draft = memory.propose(ctx, max_chars=draft_chars, min_chars=min_draft_chars)

        if draft is None:
            piece_budget = min(burst_tokens, remaining)
            r = generate_burst(upstream, api_key, model, ctx, piece_budget, timeout=timeout)
            stats.requests += 1
            if r.error:
                # Fail-safe: try to finish the rest plainly rather than truncate.
                tail, finish = _finish_plain(upstream, api_key, model, ctx, remaining,
                                             stats, timeout, first_error=r.error)
                out += tail
                stats.finish_reason = finish or stats.finish_reason
                break
            stats.burst_rounds += 1
            forced_bursts = max(0, forced_bursts - 1)
            piece = r.text
            if not piece:
                stats.finish_reason = r.finish_reason or "stop"
                break
            comp = (r.usage or {}).get("completion_tokens")
            if comp is None:
                # No trustworthy token count: neither over- nor under-counting is
                # safe (one overruns the budget, the other truncates output). Drop
                # this piece and finish the remaining budget in a single plain call
                # whose length the engine bounds exactly.
                stats.usage_fallback = True
                tail, finish = _finish_plain(upstream, api_key, model, ctx, remaining,
                                             stats, timeout)
                out += tail
                stats.finish_reason = finish or stats.finish_reason
                break
            n = min(int(comp), piece_budget)
            stats.tokens_burst += n
            stats.tokens_total += n
            ctx += piece
            out += piece
            memory.observe(piece)
            if r.finish_reason == "stop":
                stats.finish_reason = "stop"
                break
            continue

        # --- verify the draft with one scoring round-trip ------------------ #
        sent = ctx + draft
        sr = score_completion(upstream, api_key, model, sent, shift=shift, timeout=timeout)
        stats.requests += 1
        if sr.error:
            # Fail-safe: a verification error must not return a truncated
            # "success". Finish the remaining budget plainly from the last
            # confirmed token; the output still equals plain generation.
            tail, finish = _finish_plain(upstream, api_key, model, ctx, remaining,
                                         stats, timeout, first_error=sr.error)
            out += tail
            stats.finish_reason = finish or stats.finish_reason
            break
        stats.verify_rounds += 1
        draft_toks = sr.draft_tokens(len(ctx), len(sent))
        if draft_toks is None:
            # Tokenization merged characters across the seam: unverifiable.
            # A one-token boundary hiccup, not an acceptance collapse, so step
            # past it with a single burst and resume speculating.
            stats.seam_fallbacks += 1
            stats.verify_rounds_zero_accept += 1
            forced_bursts = 1
            continue

        accepted: list[str] = []
        correction: str | None = None
        for t in draft_toks:
            if t.is_greedy:
                accepted.append(t.surface)
            else:
                # Only correct when the engine's greedy choice here is itself
                # unambiguous and appendable; otherwise leave it to a burst.
                if t.top_surface and not t.top_ambiguous:
                    correction = t.top_surface
                break

        # Never emit past the budget: a plain max_tokens run stops exactly at
        # the cap, and byte-identity with it is the whole point.
        remaining = max_tokens - stats.tokens_total
        accepted = accepted[:remaining]
        new_text = "".join(accepted)
        n_new = len(accepted)
        finish = None
        whole_draft_accepted = (len(accepted) == len(draft_toks))
        if n_new < remaining and correction is not None:
            new_text += correction
            n_new += 1
            stats.tokens_correction += 1
        elif n_new < remaining and correction is None and whole_draft_accepted:
            bonus = sr.greedy_after(len(sent))
            if bonus.surface and not bonus.ambiguous:
                new_text += bonus.surface
                n_new += 1
                stats.tokens_bonus += 1
                finish = sr.finish_reason
            elif sr.finish_reason == "stop":
                finish = "stop"
        elif whole_draft_accepted and sr.finish_reason == "stop":
            finish = "stop"

        if len(accepted) == 0:
            stats.verify_rounds_zero_accept += 1
        stats.tokens_accepted += len(accepted)
        stats.tokens_total += n_new
        stats.recent_accepts.append(len(accepted))
        del stats.recent_accepts[:-8]

        if n_new == 0:
            # Nothing emitted this round (first draft token unverifiable and no
            # usable correction). Force a plain burst so we always make forward
            # progress instead of re-proposing the same wall.
            forced_bursts = 1
        elif len(stats.recent_accepts) >= 3 and sum(stats.recent_accepts[-3:]) == 0:
            # DSpark-style backoff: when drafts stop landing, leave the verify
            # lane for a while instead of paying scoring overhead for nothing.
            forced_bursts = backoff_rounds

        ctx += new_text
        out += new_text
        memory.observe(new_text)
        if finish == "stop":
            stats.finish_reason = "stop"
            break

    if stats.finish_reason is None and stats.tokens_total >= max_tokens:
        stats.finish_reason = "length"
    return out, stats


def _plain_only(upstream: str, api_key: str, model: str, prompt: str, max_tokens: int,
                stats: SpecStats, timeout: int) -> tuple[str, SpecStats]:
    """One plain generation call — what a non-accelerated client would send.

    The engine's ``max_tokens`` bounds the output length, so this is lossless and
    budget-safe without any token counting on our side.
    """
    r = generate_burst(upstream, api_key, model, prompt, max_tokens, timeout=timeout)
    stats.requests += 1
    if r.error:
        stats.error = r.error
        return "", stats
    stats.burst_rounds += 1
    comp = (r.usage or {}).get("completion_tokens")
    stats.tokens_burst = int(comp) if comp is not None else 0
    stats.tokens_total = min(int(comp), max_tokens) if comp is not None else (max_tokens if r.text else 0)
    stats.finish_reason = r.finish_reason or ("length" if r.text else "stop")
    return r.text, stats


def _finish_plain(upstream: str, api_key: str, model: str, ctx: str, remaining: int,
                  stats: SpecStats, timeout: int, first_error: str | None = None
                  ) -> tuple[str, str | None]:
    """Finish the remaining budget in a single plain call, from the last token.

    Used both when a burst/verify round errors (fail-safe: never truncate a
    "success") and when usage data is missing (budget cannot be tracked). Because
    everything emitted so far is exact greedy output, plain-continuing from ``ctx``
    yields the same bytes as a single plain ``max_tokens`` call would. Preserves
    the original error in telemetry and marks the run degraded when one occurred.
    """
    if first_error is not None:
        stats.error = stats.error or first_error
        stats.degraded_to_plain = True
    if remaining <= 0:
        return "", stats.finish_reason
    r = generate_burst(upstream, api_key, model, ctx, remaining, timeout=timeout)
    stats.requests += 1
    if r.error:
        stats.error = stats.error or r.error
        return "", stats.finish_reason
    stats.burst_rounds += 1
    comp = (r.usage or {}).get("completion_tokens")
    n = min(int(comp), remaining) if comp is not None else (remaining if r.text else 0)
    stats.tokens_burst += n
    stats.tokens_total += n
    return r.text, (r.finish_reason or ("length" if r.text else "stop"))
