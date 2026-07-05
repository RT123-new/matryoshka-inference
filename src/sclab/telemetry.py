"""Thread-safe telemetry store powering the live dashboard.

The OpenAI-compatible server updates this as tokens are generated; the dashboard
polls :meth:`TelemetryStore.snapshot` a few times a second. Everything is plain
data so the snapshot serialises straight to JSON.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Live:
    active: bool = False
    request_id: str = ""
    mode: str = ""              # "diffusion" | "ar"
    phase: str = "idle"         # "route" | "draft" | "verify" | "decode" | "stream" | "done"
    prompt_preview: str = ""
    tokens: int = 0
    started: float = 0.0
    accepted_per_pass: float = 1.0
    acceptance_rate: float = 0.0

    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started) if self.active else 0.0

    def tok_s(self) -> float:
        return (self.tokens / self.elapsed()) if self.active and self.tokens else 0.0


class TelemetryStore:
    def __init__(self, baseline_tok_s: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._live = _Live()
        self._history: deque[dict[str, Any]] = deque(maxlen=40)
        self._spark: deque[float] = deque(maxlen=160)  # rolling tok/s samples
        # aggregate totals
        self.requests = 0
        self.tokens_total = 0
        self.by_mode = {"diffusion": 0, "ar": 0, "proxy": 0}   # requests per mode
        self.tokens_by_source = {"diffusion": 0, "copy": 0, "ar": 0, "model": 0}
        self.verification_passes = 0
        self.accepted_from_draft = 0
        self.pruned_positions = 0
        # rolling per-mode throughput (tok/s) to estimate a live speedup
        self._ar_tps: deque[float] = deque(maxlen=12)
        self._diff_tps: deque[float] = deque(maxlen=12)
        self.baseline_tok_s = baseline_tok_s

    # -- generation hooks ------------------------------------------------- #
    def start(self, request_id: str, mode: str, prompt_preview: str) -> None:
        with self._lock:
            self._live = _Live(
                active=True, request_id=request_id, mode=mode,
                phase="route", prompt_preview=prompt_preview[:160],
                started=time.perf_counter(),
            )

    def phase(self, phase: str) -> None:
        with self._lock:
            self._live.phase = phase

    def tick(self, telemetry: Optional[Any] = None) -> None:
        """Called once per emitted token."""
        with self._lock:
            lv = self._live
            lv.tokens += 1
            lv.phase = "verify" if lv.mode == "diffusion" else "decode"
            if telemetry is not None:
                lv.accepted_per_pass = telemetry.accepted_tokens_per_verification_pass
                lv.acceptance_rate = telemetry.draft_acceptance_rate
            self._spark.append(round(lv.tok_s(), 1))

    def set_tokens(self, n: int) -> None:
        """Record a whole non-streaming response's token count at once."""
        with self._lock:
            self._live.tokens = n
            self._spark.append(round(self._live.tok_s(), 1))

    def finish(self, telemetry: Optional[Any], prompt_tokens: int) -> None:
        with self._lock:
            lv = self._live
            elapsed = lv.elapsed()
            tok_s = lv.tok_s()
            summary = telemetry.summary() if telemetry is not None else {}
            self.requests += 1
            self.tokens_total += lv.tokens
            self.by_mode[lv.mode] = self.by_mode.get(lv.mode, 0) + 1
            if lv.mode == "proxy":
                # External model: no draft/verify — every token is autoregressive.
                self.tokens_by_source["model"] += lv.tokens
            for src, n in (summary.get("source_mix") or {}).items():
                self.tokens_by_source[src] = self.tokens_by_source.get(src, 0) + n
            self.verification_passes += summary.get("verification_passes", lv.tokens)
            self.accepted_from_draft += int(
                summary.get("draft_acceptance_rate", 0)
                * max(0, summary.get("verification_passes", 0))
            )
            self.pruned_positions += summary.get("pruned_draft_positions", 0)
            # Only Orthrus modes feed the AR-vs-diffusion speedup estimate.
            if lv.mode == "diffusion":
                self._diff_tps.append(tok_s)
            elif lv.mode == "ar":
                self._ar_tps.append(tok_s)
            self._history.appendleft({
                "request_id": lv.request_id,
                "mode": lv.mode,
                "tokens": lv.tokens,
                "tok_s": round(tok_s, 1),
                "ms": round(elapsed * 1000),
                "accepted_per_pass": round(summary.get(
                    "accepted_tokens_per_verification_pass", lv.accepted_per_pass), 2),
                "acceptance_rate": round(summary.get(
                    "draft_acceptance_rate", lv.acceptance_rate), 3),
                "prompt_preview": lv.prompt_preview,
                "prompt_tokens": prompt_tokens,
                "pruned": summary.get("pruned_draft_positions", 0),
                "ar_lane_steps": summary.get("ar_lane_steps", 0),
            })
            self._live = _Live(phase="idle")

    # -- dashboard read --------------------------------------------------- #
    def _baseline(self) -> float:
        if self._ar_tps:
            return sum(self._ar_tps) / len(self._ar_tps)
        return self.baseline_tok_s

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lv = self._live
            diff_tps = (sum(self._diff_tps) / len(self._diff_tps)) if self._diff_tps else 0.0
            base = self._baseline()
            speedup = (diff_tps / base) if (base and diff_tps) else None
            return {
                "live": {
                    "active": lv.active,
                    "mode": lv.mode,
                    "phase": lv.phase,
                    "tokens": lv.tokens,
                    "tok_s": round(lv.tok_s(), 1),
                    "accepted_per_pass": round(lv.accepted_per_pass, 2),
                    "acceptance_rate": round(lv.acceptance_rate, 3),
                    "prompt_preview": lv.prompt_preview,
                },
                "totals": {
                    "requests": self.requests,
                    "tokens": self.tokens_total,
                    "by_mode": dict(self.by_mode),
                    "tokens_by_source": dict(self.tokens_by_source),
                    "pruned_positions": self.pruned_positions,
                },
                "rates": {
                    "diffusion_tok_s": round(diff_tps, 1),
                    "ar_tok_s": round(base, 1),
                    "speedup_vs_ar": round(speedup, 2) if speedup else None,
                },
                "spark": list(self._spark),
                "history": list(self._history),
            }
