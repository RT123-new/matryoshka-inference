"""The lossless speculation loop: draft from memory, verify by scoring.

Every emitted token is one of:

* a **burst** token — plain greedy generation from the engine (what a
  non-accelerated client would have received),
* an **accepted draft** token — verified equal to the engine's greedy choice
  at its position by a scoring round-trip,
* a **correction** token — the engine's greedy choice at the first position
  where the draft diverged,
* a **bonus** token — the single token the engine generates past a fully
  accepted draft.

By induction all four are exactly what greedy decoding would have produced,
so the final text is byte-identical to running the engine plainly — the
speedup comes purely from verifying many tokens per round-trip instead of
generating one per sequential decode step.

Seam safety: if the engine's joint tokenization of ``context + draft`` merges
characters across the boundary (or the correction/bonus token has an empty
surface, e.g. a special token), the round is discarded and the loop falls
back to a plain burst — never guess, always fall back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sclab.spec.memory import LookupMemory
from sclab.spec.verify import generate_burst, score_completion


@dataclass
class SpecStats:
    requests: int = 0
    verify_rounds: int = 0
    burst_rounds: int = 0
    tokens_total: int = 0
    tokens_accepted: int = 0     # draft tokens that survived verification
    tokens_correction: int = 0
    tokens_bonus: int = 0
    tokens_burst: int = 0
    seam_fallbacks: int = 0
    finish_reason: str | None = None
    error: str | None = None
    recent_accepts: list[int] = field(default_factory=list)

    @property
    def accepted_per_verify(self) -> float:
        if not self.verify_rounds:
            return 0.0
        return (self.tokens_accepted + self.tokens_correction + self.tokens_bonus) / self.verify_rounds

    @property
    def tokens_per_request(self) -> float:
        """The universal north-star: emitted tokens per engine round-trip.

        Plain sequential decoding is bounded by 1 token per decode step;
        anything above ~1 per *request* here is time bought back.
        """
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
    draft_chars: int = 64,
    min_draft_chars: int = 8,
    burst_tokens: int = 16,
    backoff_rounds: int = 4,
    timeout: int = 600,
) -> tuple[str, SpecStats]:
    """Generate greedily via the engine's public API, faster where possible.

    Returns ``(generated_text, stats)``. The text is byte-identical to what a
    single plain greedy request would produce (see module docstring); pass the
    same ``memory`` across calls to let acceptance compound over a session.
    """
    memory = memory if memory is not None else LookupMemory()
    stats = SpecStats()
    memory.observe(prompt)
    ctx = prompt
    out = ""
    forced_bursts = 0   # anti-thrash: after bad verify rounds, burst for a while

    while stats.tokens_total < max_tokens:
        draft = None
        if forced_bursts == 0:
            draft = memory.propose(ctx, max_chars=draft_chars, min_chars=min_draft_chars)

        if draft is None:
            piece_budget = min(burst_tokens, max_tokens - stats.tokens_total)
            r = generate_burst(upstream, api_key, model, ctx, piece_budget, timeout=timeout)
            stats.requests += 1
            if r.error:
                stats.error = r.error
                break
            stats.burst_rounds += 1
            forced_bursts = max(0, forced_bursts - 1)
            piece = r.text
            if not piece:
                stats.finish_reason = r.finish_reason or "stop"
                break
            n = int((r.usage or {}).get("completion_tokens") or 0) or max(1, len(piece.split()))
            n = min(n, piece_budget)
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
        sr = score_completion(upstream, api_key, model, sent, timeout=timeout)
        stats.requests += 1
        if sr.error:
            stats.error = sr.error
            break
        stats.verify_rounds += 1
        draft_toks = sr.draft_tokens(len(ctx), len(sent))
        if draft_toks is None:
            # Tokenization merged characters across the seam: unverifiable.
            # This is a one-token boundary hiccup, not an acceptance collapse,
            # so step past it with a single burst and resume speculating —
            # don't trigger the long DSpark-style backoff.
            stats.seam_fallbacks += 1
            forced_bursts = 1
            continue

        accepted: list[str] = []
        correction: str | None = None
        for t in draft_toks:
            if t.is_greedy:
                accepted.append(t.surface)
            else:
                correction = t.top_surface
                break

        # Never emit past the budget: a plain max_tokens run stops exactly at
        # the cap, and byte-identity with it is the whole point.
        remaining = max_tokens - stats.tokens_total
        accepted = accepted[:remaining]
        new_text = "".join(accepted)
        n_new = len(accepted)
        finish = None
        if n_new < remaining and correction is not None:
            if not correction:
                # Empty/special-token surface: cannot append faithfully as
                # text. Step past it with one burst (which reproduces it
                # natively) rather than a long backoff.
                stats.seam_fallbacks += 1
                forced_bursts = 1
                continue
            new_text += correction
            n_new += 1
            stats.tokens_correction += 1
        elif n_new < remaining and correction is None:
            bonus = sr.generated_tokens(len(sent))
            if bonus and bonus[0].surface:
                new_text += bonus[0].surface
                n_new += 1
                stats.tokens_bonus += 1
                finish = sr.finish_reason
            elif sr.finish_reason == "stop":
                finish = "stop"
        elif correction is None and sr.finish_reason == "stop":
            finish = "stop"

        stats.tokens_accepted += len(accepted)
        stats.tokens_total += n_new
        stats.recent_accepts.append(len(accepted))
        del stats.recent_accepts[:-8]
        # DSpark-style backoff: when drafts stop landing, leave the verify
        # lane for a while instead of paying scoring overhead for nothing.
        if len(stats.recent_accepts) >= 3 and sum(stats.recent_accepts[-3:]) == 0:
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
