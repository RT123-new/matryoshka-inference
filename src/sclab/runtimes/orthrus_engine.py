"""Instrumented Orthrus MLX decode engine.

This is a telemetry-carrying fork of the generation loop in
``orthrus-main/src/model_mlx.py``. The original file is left untouched; the
model *architecture* classes are imported from it, and only the decode loop is
reimplemented here so we can measure the north-star metric of this project:

    accepted_tokens_per_verification_pass

The loop is generalised so the block that the expensive autoregressive (AR)
pass verifies can come from any *proposer*:

* ``DiffusionProposer`` - the native Orthrus dual-view diffusion forward pass.
* ``CopyProposer``      - CopySpec-style suffix match over prompt+output.

Because every proposed block is verified by the exact AR pass and only the
longest AR-correct prefix is accepted, generation stays strictly lossless with
respect to greedy AR decoding regardless of which proposer produced the block.

The block length is chosen per step by a :class:`BlockPolicy`, which supports a
fixed size or an adaptive controller driven by rolling acceptance rate, logit
entropy and cheap content detection (Phase 2 of the checklist).
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Import the Orthrus MLX architecture from the sibling repo without installing
# it (its pyproject pulls torch/flash-attn which we do not want on Mac).
# --------------------------------------------------------------------------- #

def _orthrus_src_dir() -> Path:
    override = os.environ.get("ORTHRUS_SRC")
    if override:
        return Path(override)
    # Vendored copy of the Orthrus MLX architecture (MIT, see LICENSE.orthrus).
    vendored = Path(__file__).resolve().parent.parent / "vendor" / "orthrus"
    if (vendored / "model_mlx.py").exists():
        return vendored
    # Fallback: a sibling checkout of github.com/chiennv2000/orthrus
    return Path(__file__).resolve().parents[4] / "orthrus-main" / "src"


def load_orthrus(repo_id: str):
    """Load an Orthrus MLX model + tokenizer by HF repo id.

    Returns ``(model, tokenizer, model_mlx_module)``.
    """
    src = _orthrus_src_dir()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import model_mlx  # type: ignore

    model, tokenizer = model_mlx.load_model_and_tokenizer(repo_id)
    return model, tokenizer, model_mlx


def _mx():
    import mlx.core as mx  # local import so the module is importable without mlx
    return mx


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #

@dataclass
class StepRecord:
    source: str          # "diffusion" | "copy" | "ar"
    block_size: int      # tokens verified in this pass (proposal + 1)
    proposed: int        # draft tokens offered (block_size - 1)
    accepted: int        # draft tokens that survived AR verification
    emitted: int         # tokens actually appended (accepted + 1 correction)
    entropy: float       # softmax entropy of the AR logits at the reject point
    pruned: int = 0      # draft tail positions dropped by confidence pruning


@dataclass
class DecodeTelemetry:
    mode: str
    steps: list[StepRecord] = field(default_factory=list)
    tokens_generated: int = 0

    @property
    def verification_passes(self) -> int:
        return len(self.steps)

    @property
    def accepted_tokens_per_verification_pass(self) -> float:
        """North-star metric. AR baseline == 1.0 by construction."""
        if not self.steps:
            return 0.0
        return self.tokens_generated / len(self.steps)

    @property
    def draft_acceptance_rate(self) -> float:
        proposed = sum(s.proposed for s in self.steps)
        accepted = sum(s.accepted for s in self.steps)
        return (accepted / proposed) if proposed else 0.0

    def source_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {}
        for s in self.steps:
            mix[s.source] = mix.get(s.source, 0) + s.emitted
        return mix

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "tokens_generated": self.tokens_generated,
            "verification_passes": self.verification_passes,
            "accepted_tokens_per_verification_pass": round(
                self.accepted_tokens_per_verification_pass, 4
            ),
            "draft_acceptance_rate": round(self.draft_acceptance_rate, 4),
            "draft_tokens_proposed": sum(s.proposed for s in self.steps),
            "draft_tokens_accepted": sum(s.accepted for s in self.steps),
            "source_mix": self.source_mix(),
            "pruned_draft_positions": sum(s.pruned for s in self.steps),
            "ar_lane_steps": sum(1 for s in self.steps if s.source == "ar"),
        }


# --------------------------------------------------------------------------- #
# Phase 2 - block sizing policy
# --------------------------------------------------------------------------- #

_DIGIT_RE = re.compile(r"\d")
_NEGATION_RE = re.compile(r"\b(?:not|never|no|n't|neither|nor|without)\b", re.I)
# recent-output signals that we are inside easily-copied structured text
_STRUCTURED_RE = re.compile(r"[{}\[\]<>;:=]|```|\"\w+\":")


def prune_draft(probs: list[float], tau: float) -> int:
    """DSpark-style confidence pruning: how many draft positions to keep.

    Keeps the longest prefix whose cumulative survival probability stays at or
    above ``tau``; everything past that point never enters verification.
    DSpark's confidence head is trained/calibrated; here the drafter's own
    softmax probability of each chosen token is used as an uncalibrated proxy,
    which costs nothing extra since the draft logits are already in hand.
    """
    cum = 1.0
    keep = 0
    for p in probs:
        cum *= p
        if cum < tau:
            break
        keep += 1
    return keep


class BlockPolicy:
    """Chooses the block size for the next verification pass.

    mode="fixed"     -> always ``block_size``.
    mode="adaptive"  -> grow on high acceptance, shrink on low acceptance or
                        high entropy; content-aware overrides (Phase 2.4).
    mode="scheduled" -> adaptive, plus a DSpark-style speculation scheduler:
                        when rolling acceptance collapses, drop to the AR lane
                        (block size 1 = plain decode, no draft pass) for
                        ``backoff_steps`` tokens, then re-probe with a small
                        block. Makes always-on speculation safe on workloads
                        where drafting loses (free-form prose).

    A returned block size of 1 means "take a plain AR step" - the caller must
    not run a draft pass for that step.
    """

    def __init__(
        self,
        mode: str = "fixed",
        block_size: int = 8,
        min_block: int = 2,
        max_block: int = 16,
        window: int = 8,
        grow_above: float = 0.8,
        shrink_below: float = 0.4,
        entropy_cap_bits: float = 3.5,
        structured_block: int = 16,
        probe_block: int = 6,
        backoff_steps: int = 12,
    ) -> None:
        self.mode = mode
        self.base = block_size
        self.min_block = min_block
        self.max_block = max_block
        self.grow_above = grow_above
        self.shrink_below = shrink_below
        self.entropy_cap_bits = entropy_cap_bits
        # Throughput sweet spot: past ~16 the extra draft width costs more than
        # the marginal accepted tokens buy back (Phase 1b finding on M4 Max).
        self.structured_block = structured_block
        self.probe_block = probe_block
        self.backoff_steps = backoff_steps
        self._acc: deque[float] = deque(maxlen=window)
        self._cur = block_size
        self._backoff = 0

    def _rolling(self) -> float | None:
        if not self._acc:
            return None
        return sum(self._acc) / len(self._acc)

    def next_block_size(
        self, recent_text: str = "", last_entropy: float = 0.0, in_think: bool = False
    ) -> int:
        if self.mode == "fixed":
            return max(2, self.base)

        # Scheduler AR lane: sit out speculation entirely, then re-probe small.
        if self.mode == "scheduled" and self._backoff > 0:
            self._backoff -= 1
            if self._backoff == 0:
                self._cur = self.probe_block
            return 1

        cur = self._cur
        rate = self._rolling()
        if rate is not None:
            # Grow toward the throughput sweet spot, not to max_block, so we
            # stop paying for draft width the model won't accept.
            if rate >= self.grow_above:
                cur = min(self.structured_block, cur + 2)
            elif rate <= self.shrink_below:
                cur = max(self.min_block, cur - 2)

        # Entropy gate: if the model is locally uncertain, do not over-commit.
        if last_entropy >= self.entropy_cap_bits:
            cur = max(self.min_block, min(cur, 4))

        # Content-aware overrides (Phase 2.4): structured/boilerplate text drafts
        # long, but capped at the sweet spot rather than max_block.
        if recent_text and _STRUCTURED_RE.search(recent_text):
            cur = min(self.max_block, self.structured_block)

        # Phase 5.3 thinking-lane throttle: inside a <think> span only the
        # conclusion matters, so draft at the sweet spot regardless.
        if in_think:
            cur = max(cur, min(self.max_block, self.structured_block))

        self._cur = max(2, min(self.max_block, cur))
        return self._cur

    def update(self, accepted: int, proposed: int) -> None:
        if proposed > 0:
            self._acc.append(accepted / proposed)
            # Scheduler: collapse in rolling acceptance -> leave the spec lane.
            if (
                self.mode == "scheduled"
                and self._backoff == 0
                and len(self._acc) >= 3
                and (sum(self._acc) / len(self._acc)) <= self.shrink_below
            ):
                self._backoff = self.backoff_steps
                self._acc.clear()


# --------------------------------------------------------------------------- #
# Phase 5.6 - Matryoshka mode router
# --------------------------------------------------------------------------- #

# Prompts that tend to produce highly-acceptable (structured / predictable)
# output where diffusion decoding wins on Apple Silicon.
_DIFFUSION_CUES = re.compile(
    r"\b(json|yaml|xml|html|csv|sql|code|function|class|method|array|object|"
    r"schema|table|list|dict|regex|api|config|template|boilerplate|"
    r"step[- ]?by[- ]?step|steps|calculate|compute|solve|derive|proof)\b",
    re.I,
)
# Prompts that tend to produce free-form prose where the AR baseline is faster.
_PROSE_CUES = re.compile(
    r"\b(explain|describe|essay|story|poem|discuss|opinion|summar(?:y|ise|ize)|"
    r"brainstorm|imagine|persuade|reflect|narrat)\w*\b",
    re.I,
)


def route_mode(prompt_text: str) -> tuple[str, str]:
    """Pick a decode mode for a request from its prompt shape.

    Returns ``(mode, reason)`` where mode is "diffusion" or "ar". Structured /
    reasoning prompts -> diffusion (high acceptance, real speedup); free-form
    prose -> ar (avoids the two-forward-pass overhead that regresses there).
    Ties and cue-less prompts default to ar, the safe choice.
    """
    d = len(_DIFFUSION_CUES.findall(prompt_text))
    p = len(_PROSE_CUES.findall(prompt_text))
    if d > 0 and d >= p:
        return "diffusion", f"structured/reasoning cues={d} prose cues={p}"
    return "ar", f"structured/reasoning cues={d} prose cues={p}"


# --------------------------------------------------------------------------- #
# Phase 3 - copy proposer (CopySpec-style suffix match)
# --------------------------------------------------------------------------- #

class CopyProposer:
    """Proposes a draft block by copying tokens that followed an earlier
    occurrence of the current suffix in prompt+generated history.

    Uses a rolling n-gram index (hash of the last ``ngram`` tokens -> list of
    end positions). Cheap, allocation-light, and fully sound because the AR
    pass verifies whatever it proposes.
    """

    def __init__(self, ngram: int = 4, min_match: int = 6) -> None:
        self.ngram = ngram
        self.min_match = min_match
        self._index: dict[tuple, list[int]] = {}
        self._tokens: list[int] = []

    def _key(self, i: int) -> tuple:
        return tuple(self._tokens[i - self.ngram + 1 : i + 1])

    def extend(self, new_tokens: list[int]) -> None:
        for t in new_tokens:
            self._tokens.append(t)
            i = len(self._tokens) - 1
            if i + 1 >= self.ngram:
                self._index.setdefault(self._key(i), []).append(i)

    def propose(self, want: int) -> list[int] | None:
        n = len(self._tokens)
        if n < self.ngram or want <= 0:
            return None
        key = tuple(self._tokens[n - self.ngram :])
        cands = self._index.get(key)
        if not cands:
            return None
        # Most recent earlier occurrence; verify the match extends >= min_match.
        for end in reversed(cands):
            if end == n - 1:
                continue
            match_len = self._backmatch(end, n - 1)
            if match_len >= self.min_match:
                start = end + 1
                block = self._tokens[start : start + want]
                if block:
                    return block
        return None

    def _backmatch(self, a: int, b: int) -> int:
        length = 0
        while a >= 0 and b >= 0 and self._tokens[a] == self._tokens[b]:
            length += 1
            a -= 1
            b -= 1
        return length


# --------------------------------------------------------------------------- #
# Decode loops
# --------------------------------------------------------------------------- #

def _cache_len(prompt_len: int, max_tokens: int) -> int:
    """Ring-cache capacity that safely holds prompt + generation.

    The Orthrus RingKVCache defaults to 2048 slots and silently wraps past that,
    corrupting long-context output. Size it to the actual need plus margin.
    """
    return prompt_len + max_tokens + 64


def _entropy_bits(logits_row) -> float:
    mx = _mx()
    p = mx.softmax(logits_row.astype(mx.float32), axis=-1)
    logp = mx.log(p + 1e-12)
    ent = float(-(p * logp).sum().item()) / math.log(2.0)
    return ent


def ar_generate(model, prompt_tokens, eos_token_id, max_tokens=256, temperature=0.0):
    """Pure autoregressive baseline over the same weights/KV cache.

    Yields ``(token, telemetry)`` where telemetry is finalised on the last
    yield. accepted_tokens_per_verification_pass == 1.0 by construction.
    """
    mx = _mx()
    sys.path.insert(0, str(_orthrus_src_dir()))
    import model_mlx  # type: ignore

    tel = DecodeTelemetry(mode="ar")
    tokens = list(prompt_tokens)
    cap = _cache_len(len(tokens), max_tokens)
    cache = [model_mlx.RingKVCache(max_seq_len=cap) for _ in range(len(model.model.layers))]

    logits = model(mx.array([tokens]), cache=cache, is_diffusion_pass=False)
    token = model_mlx.sample(logits[:, -1, :], temperature).item()
    tokens.append(token)
    tel.tokens_generated += 1
    tel.steps.append(StepRecord("ar", 1, 0, 0, 1, 0.0))
    yield token, tel
    if token == eos_token_id:
        return

    while tel.tokens_generated < max_tokens:
        logits = model(mx.array([[tokens[-1]]]), cache=cache, is_diffusion_pass=False)
        token = model_mlx.sample(logits[:, -1, :], temperature).item()
        tokens.append(token)
        tel.tokens_generated += 1
        tel.steps.append(StepRecord("ar", 1, 0, 0, 1, 0.0))
        yield token, tel
        if token == eos_token_id:
            return


def _think_state(prev: bool, text: str) -> bool:
    """Track whether generation is inside a <think> span from emitted text."""
    opened = text.rfind("<think>")
    closed = text.rfind("</think>")
    if opened == -1 and closed == -1:
        return prev
    return opened > closed


def orthrus_generate(
    model,
    prompt_tokens,
    eos_token_id,
    max_tokens: int = 256,
    temperature: float = 0.0,
    policy: BlockPolicy | None = None,
    copy_proposer: CopyProposer | None = None,
    detokenize: Callable[[list[int]], str] | None = None,
    prune_tau: float | None = None,
):
    """Instrumented dual-view (+ optional copy) speculative decode.

    Yields ``(token, telemetry)``. Strictly lossless vs greedy AR at
    temperature 0 because the AR pass verifies every proposed block.

    ``prune_tau`` enables DSpark-style confidence pruning: draft positions past
    the point where the drafter's cumulative confidence drops below tau are cut
    before verification, so low-value guesses never widen the verify pass.
    """
    mx = _mx()
    sys.path.insert(0, str(_orthrus_src_dir()))
    import model_mlx  # type: ignore

    mask_id = model.config.mask_token_id
    policy = policy or BlockPolicy(mode="fixed", block_size=model.config.block_size)
    tel = DecodeTelemetry(mode="diffusion")

    tokens = list(prompt_tokens)
    cap = _cache_len(len(tokens), max_tokens)
    cache = [model_mlx.RingKVCache(max_seq_len=cap) for _ in range(len(model.model.layers))]

    # Prefill + first real token (normal AR).
    logits = model(mx.array([tokens]), cache=cache, is_diffusion_pass=False)
    token = model_mlx.sample(logits[:, -1, :], temperature).item()
    tokens.append(token)
    tel.tokens_generated += 1
    if copy_proposer is not None:
        copy_proposer.extend([token])
    in_think = _think_state(False, detokenize([token])) if detokenize else False
    yield token, tel
    if token == eos_token_id:
        return

    last_entropy = 0.0
    while tel.tokens_generated < max_tokens:
        recent = detokenize(tokens[-24:]) if detokenize else ""
        bs = policy.next_block_size(
            recent_text=recent, last_entropy=last_entropy, in_think=in_think
        )
        # A block of size bs emits at most bs tokens (accepted + 1 correction),
        # so cap it at the remaining budget or generation overshoots max_tokens.
        bs = min(bs, max_tokens - tel.tokens_generated)

        # --- scheduler AR lane: plain decode step, no draft pass ------------ #
        if bs <= 1:
            logits = model(mx.array([[tokens[-1]]]), cache=cache, is_diffusion_pass=False)
            token = model_mlx.sample(logits[:, -1, :], temperature).item()
            # Entropy costs a full-vocab softmax + GPU sync; only pay for it on
            # the last backoff step, where it informs the next block choice.
            if getattr(policy, "_backoff", 0) <= 1:
                last_entropy = _entropy_bits(logits[0, -1, :])
            tokens.append(token)
            tel.tokens_generated += 1
            tel.steps.append(StepRecord("ar", 1, 0, 0, 1, last_entropy))
            if copy_proposer is not None:
                copy_proposer.extend([token])
            if detokenize:
                in_think = _think_state(in_think, detokenize([token]))
            yield token, tel
            if token == eos_token_id:
                return
            continue

        want = bs - 1

        # --- proposer: copy first, else diffusion --------------------------- #
        source = "diffusion"
        pruned = 0
        proposed = None
        if copy_proposer is not None:
            proposed = copy_proposer.propose(want)
            if proposed is not None:
                source = "copy"

        if proposed is None:
            diff_block = mx.array([[tokens[-1]] + [mask_id] * want])
            diff_logits = model(diff_block, cache=cache, is_diffusion_pass=True)
            diff_tokens = model_mlx.sample(diff_logits[:, :-1, :], temperature)
            mx.eval(diff_tokens)
            proposed_list = diff_tokens[0].tolist()
            if prune_tau is not None and proposed_list:
                dp = mx.softmax(diff_logits[0, :-1, :].astype(mx.float32), axis=-1)
                chosen = mx.take_along_axis(dp, diff_tokens[0][:, None], axis=-1)
                probs = chosen[:, 0].tolist()
                keep = prune_draft(probs, prune_tau)
                pruned = len(proposed_list) - keep
                proposed_list = proposed_list[:keep]
            proposed_arr = mx.array([proposed_list]) if proposed_list else None
        else:
            proposed_list = proposed[:want]
            proposed_arr = mx.array([proposed_list])

        # --- verify with the exact AR pass ---------------------------------- #
        last_tok = mx.array([[tokens[-1]]])
        if proposed_arr is not None:
            verify_input = mx.concatenate([last_tok, proposed_arr], axis=1)
        else:
            verify_input = last_tok
        ar_logits = model(verify_input, cache=cache, is_diffusion_pass=False)
        ar_tokens = model_mlx.sample(ar_logits, temperature)
        mx.eval(ar_logits, ar_tokens)

        d_list = proposed_list
        t_list = ar_tokens[0].tolist()

        accepted = 0
        # zip stops at the shorter list on purpose: t_list has one extra
        # element (the correction token past the last draft position).
        for d, t in zip(d_list, t_list[:-1], strict=False):
            if d == t:
                accepted += 1
            else:
                break

        new_tokens = d_list[:accepted] + [t_list[accepted]]
        last_entropy = _entropy_bits(ar_logits[0, accepted, :])

        trim_amount = verify_input.shape[1] - (accepted + 1)
        if trim_amount > 0:
            for c in cache:
                c.trim(trim_amount)

        tel.steps.append(
            StepRecord(
                source=source,
                block_size=bs,
                proposed=len(d_list),
                accepted=accepted,
                emitted=len(new_tokens),
                entropy=last_entropy,
                pruned=pruned,
            )
        )
        # Judge the drafter on the width it was ASKED to draft, not the
        # post-prune width: pruning must not mask low confidence from the
        # scheduler, or the AR-lane backoff never triggers.
        if source == "diffusion":
            policy.update(accepted, want)
        elif d_list:
            policy.update(accepted, len(d_list))
        if copy_proposer is not None:
            copy_proposer.extend(new_tokens)
        if detokenize:
            in_think = _think_state(in_think, detokenize(new_tokens))

        for t in new_tokens:
            tokens.append(t)
            tel.tokens_generated += 1
            yield t, tel
            if t == eos_token_id or tel.tokens_generated >= max_tokens:
                return
