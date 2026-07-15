"""Token-ID verified speculation: the unconditional-equivalence correctness gate.

Unlike text-surface mode, the gate here is on **token ids**, not decoded strings:

    spec.token_ids == baseline.token_ids
    spec.text_bytes == baseline.text_bytes

Text equality alone is deliberately *not* the gate — these tests include cases
where distinct ids share a surface and where decoded output re-tokenizes
differently, exactly the situations text mode cannot prove correct.

Backends are deterministic fakes exposing ids and a raw-argmax oracle, so
everything is checkable to the id without weights or a GPU:

* ``LagTokenBackend`` — a canonical char/merge tokenizer + a lag-over-ids greedy
  model (self-repetitive, so the text proposer lands drafts).
* ``PlaybackBackend`` — emits a scripted id sequence (position-only argmax), so a
  test can script duplicate-surface ids, non-canonical merges and EOS exactly.
"""

from __future__ import annotations

import pytest

from sclab.spec.backend import (
    DETERMINISTIC_POLICY,
    TOKEN_ID_NONDETERMINISTIC,
    TOKEN_ID_VERIFIED,
    DraftVerification,
    GenerationResult,
    VerificationCapability,
)
from sclab.spec.memory import LookupMemory
from sclab.spec.token_verify import (
    _propose_draft_ids,
    resolve_draft,
    spec_generate_tokens,
)

# --------------------------------------------------------------------------- #
# Deterministic fake tokenizer + backends.
# --------------------------------------------------------------------------- #


class Tokenizer:
    """Greedy-longest-match char tokenizer with optional merges and dup ids."""
    BOS, EOS = 1, 2

    def __init__(self, merges=(), dup_surfaces=()):
        self._id2b = {self.BOS: b"", self.EOS: b""}
        self._surf2id: dict[str, int] = {}
        self._next = 10
        for s in list(merges) + [chr(c) for c in range(32, 127)]:
            if s not in self._surf2id:
                self._reg(s)
        # An extra id whose bytes equal an existing surface: two ids, one surface.
        self.dup: dict[str, int] = {}
        for s in dup_surfaces:
            i = self._newid()
            self._id2b[i] = s.encode("utf-8")
            self.dup[s] = i

    def _newid(self) -> int:
        i = self._next
        self._next += 1
        return i

    def _reg(self, s: str) -> int:
        i = self._newid()
        self._id2b[i] = s.encode("utf-8")
        self._surf2id[s] = i
        return i

    def id_of(self, surf: str) -> int:
        return self._surf2id[surf]

    def encode(self, text: str) -> list[int]:
        ids = [self.BOS]
        keys = sorted(self._surf2id, key=len, reverse=True)
        i = 0
        while i < len(text):
            for k in keys:
                if text.startswith(k, i):
                    ids.append(self._surf2id[k])
                    i += len(k)
                    break
            else:
                # Auto-register any unseen character (e.g. Unicode) as a token.
                c = text[i]
                self._reg(c)
                keys = sorted(self._surf2id, key=len, reverse=True)
                ids.append(self._surf2id[c])
                i += 1
        return ids

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self._id2b[i] for i in ids)


def _cap(tok: Tokenizer) -> VerificationCapability:
    return VerificationCapability(TOKEN_ID_VERIFIED, deterministic=True, supports_bonus=True,
                                  eos_token_id=tok.EOS, policy=dict(DETERMINISTIC_POLICY))


class LagTokenBackend:
    """Canonical tokenizer + greedy model: next id == the id ``lag`` back."""

    def __init__(self, tok: Tokenizer, lag: int = 8, seed: str = "x"):
        self.tok = tok
        self.lag = lag
        self.seed_id = tok.id_of(seed) if seed in tok._surf2id else tok.encode(seed)[-1]

    def capability(self):
        return _cap(self.tok)

    def encode_context(self, text):
        return self.tok.encode(text)

    def decode_tokens(self, ids):
        return self.tok.decode(ids)

    def _argmax(self, seq):
        body = [i for i in seq if i not in (self.tok.BOS, self.tok.EOS)]
        return body[-self.lag] if len(body) >= self.lag else self.seed_id

    def generate_plain(self, ctx, max_tokens):
        seq = list(ctx)
        out = []
        for _ in range(max_tokens):
            nid = self._argmax(seq)
            if nid == self.tok.EOS:
                return GenerationResult(out, "stop")
            out.append(nid)
            seq.append(nid)
        return GenerationResult(out, "length")

    def verify_draft(self, ctx, draft):
        preds = [self._argmax(list(ctx) + list(draft[:j])) for j in range(len(draft) + 1)]
        return DraftVerification(preds)


class PlaybackBackend:
    """Deterministic model emitting a fixed id script (position-only argmax)."""

    def __init__(self, tok: Tokenizer, prompt: str, script: list[int]):
        self.tok = tok
        self.script = list(script)
        self._base = len(tok.encode(prompt))

    def capability(self):
        return _cap(self.tok)

    def encode_context(self, text):
        return self.tok.encode(text)

    def decode_tokens(self, ids):
        return self.tok.decode(ids)

    def _gi(self, ctx):
        return len(ctx) - self._base

    def generate_plain(self, ctx, max_tokens):
        gi = self._gi(ctx)
        out = []
        for j in range(max_tokens):
            k = gi + j
            if k >= len(self.script):
                return GenerationResult(out, "length")
            if self.script[k] == self.tok.EOS:
                return GenerationResult(out, "stop")
            out.append(self.script[k])
        return GenerationResult(out, "length")

    def verify_draft(self, ctx, draft):
        gi = self._gi(ctx)
        preds = [self.script[gi + j] if gi + j < len(self.script) else self.tok.EOS
                 for j in range(len(draft) + 1)]
        return DraftVerification(preds)


def _baseline(backend, prompt, max_tokens):
    ids = backend.encode_context(prompt)
    gen = backend.generate_plain(ids, max_tokens)
    return gen.token_ids, backend.decode_tokens(gen.token_ids)


def _assert_equal_to_baseline(backend, prompt, max_tokens, **kw):
    base_ids, base_bytes = _baseline(backend, prompt, max_tokens)
    g = spec_generate_tokens(backend, prompt, max_tokens=max_tokens, capability=backend.capability(), **kw)
    assert g.token_ids == base_ids, (g.token_ids, base_ids)     # THE gate: id identity
    assert g.text_bytes == base_bytes                            # bytes follow from ids
    assert len(g.token_ids) <= max_tokens
    return g


# --------------------------------------------------------------------------- #
# resolve_draft — accept/correct/bonus logic (cases 1–4).
# --------------------------------------------------------------------------- #

def test_fully_correct_draft_accepted_then_bonus():
    r = resolve_draft([10, 11, 12], [10, 11, 12, 13], remaining=100)
    assert r.emitted_ids == [10, 11, 12, 13]
    assert r.n_accepted == 3 and r.n_bonus == 1 and r.all_accepted


def test_first_token_mismatch_corrected():
    r = resolve_draft([10, 11, 12], [99, 11, 12, 13], remaining=100)
    assert r.emitted_ids == [99]
    assert r.n_accepted == 0 and r.n_correction == 1


def test_middle_token_mismatch_accepts_prefix_and_corrects():
    r = resolve_draft([10, 11, 12], [10, 99, 12, 13], remaining=100)
    assert r.emitted_ids == [10, 99]
    assert r.n_accepted == 1 and r.n_correction == 1


def test_no_token_is_skipped_between_accept_and_correction():
    # The emitted ids are exactly the accepted prefix followed by the correction,
    # with nothing dropped in between.
    r = resolve_draft([10, 11, 12, 13], [10, 11, 77, 13, 14], remaining=100)
    assert r.emitted_ids == [10, 11, 77]


# --------------------------------------------------------------------------- #
# Exact budget + EOS (cases 6, 7).
# --------------------------------------------------------------------------- #

REPEAT = "the quick brown fox jumps over the lazy dog and then the"


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 7, 16, 50, 120])
def test_exact_max_token_budget(max_tokens):
    b = LagTokenBackend(Tokenizer(), lag=8)
    g = _assert_equal_to_baseline(b, REPEAT, max_tokens, draft_chars=96, burst_tokens=8)
    assert g.stats.tokens_total <= max_tokens


def test_eos_stops_at_the_same_place_as_plain():
    tok = Tokenizer()
    prompt = "abcdefgh"
    # script: emit 5 real tokens, then EOS. Plain and spec must both stop there.
    script = [tok.id_of(c) for c in "ijklm"] + [tok.EOS] + [tok.id_of(c) for c in "nop"]
    b = PlaybackBackend(tok, prompt, script)
    base_ids, base_bytes = _baseline(b, prompt, 50)
    assert base_ids == [tok.id_of(c) for c in "ijklm"]      # stopped before EOS
    g = spec_generate_tokens(b, prompt, max_tokens=50, capability=b.capability())
    assert g.token_ids == base_ids and g.text_bytes == base_bytes
    assert g.stats.finish_reason == "stop"


def test_warm_memory_lands_drafts_and_stays_exact():
    b = LagTokenBackend(Tokenizer(), lag=10)
    mem = LookupMemory()
    base_ids, _ = _baseline(b, REPEAT, 200)
    mem.observe(REPEAT + b.decode_tokens(base_ids).decode())
    g = _assert_equal_to_baseline(b, REPEAT, 150, memory=mem, draft_chars=96, burst_tokens=8)
    assert g.stats.draft_ids_accepted > 0
    assert g.stats.draft_ids_accepted_per_verify > 1.0
    # telemetry is present and disaggregated
    s = g.stats.summary()
    for key in ("draft_text_proposals", "draft_ids_proposed", "draft_ids_accepted",
                "correction_ids", "bonus_ids", "token_id_verify_rounds",
                "text_surface_verify_rounds", "context_prefix_mismatch"):
        assert key in s


# --------------------------------------------------------------------------- #
# Duplicate surfaces (case 8) and non-canonical re-tokenization (case 9):
# text equality would be fooled; token-id mode stays exact.
# --------------------------------------------------------------------------- #

def _text_roundtrip_would_diverge(tok, ids):
    """True if re-tokenizing the decoded output yields a *different* id sequence."""
    return tok.encode(tok.decode(ids).decode("utf-8"))[1:] != ids


def test_duplicate_decoded_surface_with_distinct_ids():
    tok = Tokenizer(dup_surfaces=["X"])
    dup = tok.dup["X"]
    canon = tok.id_of("X")
    assert dup != canon
    assert tok.decode([dup]) == tok.decode([canon])       # identical surface
    prompt = "value X here value X here value X here "
    # The model replays the prompt but emits the DUP id for X, not the canonical.
    body = tok.encode(prompt)[1:]
    body = [dup if i == canon else i for i in body]
    b = PlaybackBackend(tok, prompt, body * 4)

    base_ids, base_bytes = _baseline(b, prompt, 120)
    assert dup in base_ids
    mem = LookupMemory()
    mem.observe(prompt + b.decode_tokens(base_ids).decode())
    g = spec_generate_tokens(b, prompt, max_tokens=120, capability=b.capability(),
                             memory=mem, draft_chars=96, burst_tokens=8)
    # THE gate: exact id identity, even though the surface hides the id.
    assert g.token_ids == base_ids
    assert g.text_bytes == base_bytes
    assert dup in g.token_ids
    # A text-surface reconstruction of the very same output would re-tokenize the
    # dup surface to the canonical id — i.e. lose information token mode kept.
    assert _text_roundtrip_would_diverge(tok, g.token_ids)


def test_noncanonical_reencode_stays_exact_in_token_mode():
    tok = Tokenizer(merges=["ab"])
    a, b_id, ab = tok.id_of("a"), tok.id_of("b"), tok.id_of("ab")
    prompt = "start "
    # Model emits the two single-char ids a,b — which decode to "ab" and would
    # re-tokenize to the single merge id [ab]. Repeat so there is plenty to check.
    script = ([a, b_id, tok.id_of(" ")] * 20)
    backend = PlaybackBackend(tok, prompt, script)

    base_ids, base_bytes = _baseline(backend, prompt, 60)
    assert a in base_ids and b_id in base_ids and ab not in base_ids
    mem = LookupMemory()
    mem.observe(prompt + backend.decode_tokens(base_ids).decode())
    g = spec_generate_tokens(backend, prompt, max_tokens=60, capability=backend.capability(),
                             memory=mem, draft_chars=96, burst_tokens=8)
    assert g.token_ids == base_ids                     # exact ids: a,b never merged
    assert g.text_bytes == base_bytes
    assert _text_roundtrip_would_diverge(tok, g.token_ids)


def test_noncanonical_draft_proposal_is_rejected_as_prefix_mismatch():
    # A proposed draft whose re-tokenization does not reproduce the authoritative
    # context ids as a prefix is rejected (seam/canonicalization failure).
    tok = Tokenizer(merges=["ab"])
    backend = LagTokenBackend(tok, lag=8)
    context_text = "zzzzzzzzzzzzzzza"              # >= 16 chars, ends in a bare 'a'
    context_ids = tok.encode(context_text)
    assert context_ids[-1] == tok.id_of("a")
    mem = LookupMemory()
    # After this exact 16-char suffix the memory has seen "bcdefgh", so the
    # proposal is "bcdefgh"; the seam "a" + "b" re-tokenizes into the merge id
    # 'ab', which no longer reproduces the authoritative context ids as a prefix.
    mem.observe(context_text + "bcdefgh")
    from sclab.spec.token_verify import TokenSpecStats
    stats = TokenSpecStats()
    draft = _propose_draft_ids(backend, context_ids, mem, draft_chars=16, min_draft_chars=4, stats=stats)
    assert draft is None
    assert stats.draft_text_proposals == 1          # a text draft WAS proposed...
    assert stats.context_prefix_mismatch == 1       # ...but rejected at the seam
    assert stats.draft_tokenization_rejected == 1


# --------------------------------------------------------------------------- #
# Authoritative context stays ids (case 10); wrong indexing fails loudly (11).
# --------------------------------------------------------------------------- #

class RecordingBackend:
    """Wraps a backend and records every context passed into the verify lane."""

    def __init__(self, inner, prompt):
        self.inner = inner
        self.prompt_ids = inner.encode_context(prompt)
        self.seen: list[list[int]] = []

    def capability(self):
        return self.inner.capability()

    def encode_context(self, text):
        return self.inner.encode_context(text)

    def decode_tokens(self, ids):
        return self.inner.decode_tokens(ids)

    def generate_plain(self, ctx, max_tokens):
        self.seen.append(list(ctx))
        return self.inner.generate_plain(ctx, max_tokens)

    def verify_draft(self, ctx, draft):
        self.seen.append(list(ctx))
        return self.inner.verify_draft(ctx, draft)


def test_context_ids_are_threaded_not_reconstructed_by_retokenizing():
    tok = Tokenizer(merges=["ab"])
    a, b_id = tok.id_of("a"), tok.id_of("b")
    prompt = "start "
    inner = PlaybackBackend(tok, prompt, [a, b_id, tok.id_of(" ")] * 20)
    rec = RecordingBackend(inner, prompt)
    base_ids, _ = _baseline(inner, prompt, 40)
    g = spec_generate_tokens(rec, prompt, max_tokens=40, capability=rec.capability())
    assert g.token_ids == base_ids
    # Every context handed to the engine keeps the authoritative prompt prefix...
    assert all(ctx[: len(rec.prompt_ids)] == rec.prompt_ids for ctx in rec.seen)
    # ...and at least one context is non-canonical (a,b would re-tokenize to [ab]),
    # which can only be true if the loop threaded ids rather than re-encoding text.
    assert any(_text_roundtrip_would_diverge(tok, ctx[len(rec.prompt_ids):])
               for ctx in rec.seen if len(ctx) > len(rec.prompt_ids))


class WrongIndexBackend(LagTokenBackend):
    """Deliberately reads the argmax one position too late — a wiring bug."""

    def verify_draft(self, ctx, draft):
        # Correct code uses prefixes ctx+draft[:j]; this uses ctx+draft[:j+1],
        # i.e. predicts token j+1 where it should predict token j.
        preds = [self._argmax(list(ctx) + list(draft[:j + 1])) for j in range(len(draft) + 1)]
        return DraftVerification(preds)


def test_wrong_prediction_indexing_diverges_loudly():
    tok = Tokenizer()
    good = LagTokenBackend(tok, lag=10)
    bad = WrongIndexBackend(tok, lag=10)
    mem_g, mem_b = LookupMemory(), LookupMemory()
    base_ids, _ = _baseline(good, REPEAT, 200)
    primer = REPEAT + good.decode_tokens(base_ids).decode()
    mem_g.observe(primer)
    mem_b.observe(primer)
    exp_ids, _ = _baseline(good, REPEAT, 120)
    base = spec_generate_tokens(good, REPEAT, max_tokens=120, capability=good.capability(),
                                memory=mem_g, draft_chars=96, burst_tokens=8)
    wrong = spec_generate_tokens(bad, REPEAT, max_tokens=120, capability=bad.capability(),
                                 memory=mem_b, draft_chars=96, burst_tokens=8)
    assert base.token_ids == exp_ids                    # correct wiring is exact
    assert wrong.token_ids != exp_ids                   # wrong wiring is caught


# --------------------------------------------------------------------------- #
# Runtime verification failure degrades to plain generation (case 12).
# --------------------------------------------------------------------------- #

class FlakyVerifyBackend(LagTokenBackend):
    """Raises a verify error after the first verify round."""

    def __init__(self, tok, lag=10):
        super().__init__(tok, lag=lag)
        self._verifies = 0

    def verify_draft(self, ctx, draft):
        self._verifies += 1
        if self._verifies >= 2:
            return DraftVerification(error="simulated backend verify failure")
        return super().verify_draft(ctx, draft)


def test_verification_failure_degrades_to_plain_without_corrupting():
    tok = Tokenizer()
    good = LagTokenBackend(tok, lag=10)
    flaky = FlakyVerifyBackend(tok, lag=10)
    base_ids, base_bytes = _baseline(good, REPEAT, 200)
    mem = LookupMemory()
    mem.observe(REPEAT + good.decode_tokens(base_ids).decode())
    g = spec_generate_tokens(flaky, REPEAT, max_tokens=150, capability=flaky.capability(),
                             memory=mem, draft_chars=96, burst_tokens=8)
    exp_ids, exp_bytes = _baseline(good, REPEAT, 150)
    assert g.token_ids == exp_ids               # output equals plain generation
    assert g.text_bytes == exp_bytes
    assert g.stats.degraded_to_plain is True
    assert g.stats.error and "verify failure" in g.stats.error


# --------------------------------------------------------------------------- #
# Unusable / nondeterministic capability → plain generation only.
# --------------------------------------------------------------------------- #

def test_nondeterministic_capability_is_not_usable():
    cap = VerificationCapability(TOKEN_ID_NONDETERMINISTIC, deterministic=False)
    assert not cap.usable


def test_unusable_capability_runs_plain_and_stays_exact():
    tok = Tokenizer()
    b = LagTokenBackend(tok, lag=8)
    base_ids, base_bytes = _baseline(b, REPEAT, 40)
    unusable = VerificationCapability(TOKEN_ID_NONDETERMINISTIC, deterministic=False,
                                      eos_token_id=tok.EOS)
    g = spec_generate_tokens(b, REPEAT, max_tokens=40, capability=unusable)
    assert g.token_ids == base_ids and g.text_bytes == base_bytes
    assert g.stats.spec_available is False
    assert g.stats.token_id_verify_rounds == 0


# --------------------------------------------------------------------------- #
# Unicode and seams (token mode): id + byte identity across tricky text.
# --------------------------------------------------------------------------- #

UNICODE_PROMPTS = [
    "plain ascii repeated plain ascii repeated plain ascii",
    "café au lait café au lait café au lait café au lait",
    "Shqipëria është e bukur Shqipëria është e bukur Shqipëria",
    "😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀 😀 🎉 🚀",
    "“curly” — em–dash “curly” — em–dash “curly” — em–dash",
    "é à ô é à ô é à ô",
    "   leading spaces    leading spaces    leading spaces",
    "trailing spaces    trailing spaces    trailing spaces    ",
    "line\n\n\nbreaks line\n\n\nbreaks line\n\n\nbreaks line",
    "punct!!! ??? ... punct!!! ??? ... punct!!! ??? ...",
]


@pytest.mark.parametrize("prompt", UNICODE_PROMPTS, ids=lambda p: repr(p[:16]))
@pytest.mark.parametrize("max_tokens", [16, 64])
def test_unicode_and_seam_token_identity(prompt, max_tokens):
    b = LagTokenBackend(Tokenizer(), lag=8)
    mem = LookupMemory()
    base_ids, _ = _baseline(b, prompt, 200)
    mem.observe(prompt + b.decode_tokens(base_ids).decode())
    _assert_equal_to_baseline(b, prompt, max_tokens, memory=mem, draft_chars=64, burst_tokens=8)


def test_token_merge_across_context_draft_boundary_stays_exact():
    # A merge token 'ab' spanning the context/draft seam must never corrupt output:
    # the proposal is rejected at the seam and a burst makes exact progress.
    tok = Tokenizer(merges=["ab"])
    b = LagTokenBackend(tok, lag=6)
    prompt = "xyzab xyzab xyzab xyzab xyzab "
    mem = LookupMemory()
    base_ids, _ = _baseline(b, prompt, 200)
    mem.observe(prompt + b.decode_tokens(base_ids).decode())
    _assert_equal_to_baseline(b, prompt, 80, memory=mem, draft_chars=96, burst_tokens=8)
