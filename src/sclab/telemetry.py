"""Thread-safe telemetry store powering the live dashboard.

The OpenAI-compatible server updates this as tokens are generated; the dashboard
polls :meth:`TelemetryStore.snapshot` a few times a second. Everything is plain
data so the snapshot serialises straight to JSON.

Sessions are keyed by request id, so overlapping requests (the proxy backend
serves them concurrently) each keep their own token counts and timings instead
of trampling a single shared slot. The dashboard's "live" view follows the most
recently started request that is still running.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Any


class _Live:
    __slots__ = ("request_id", "mode", "phase", "prompt_preview", "tokens", "started")

    def __init__(self, request_id: str = "", mode: str = "", prompt_preview: str = "") -> None:
        self.request_id = request_id
        self.mode = mode                    # "diffusion" | "ar" | "proxy"
        self.phase = "route"                # "route" | "draft" | "verify" | "decode" | "stream"
        self.prompt_preview = prompt_preview[:160]
        self.tokens = 0
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started)

    def tok_s(self) -> float:
        return (self.tokens / self.elapsed()) if self.tokens else 0.0


class TelemetryStore:
    def __init__(self, baseline_tok_s: float = 0.0) -> None:
        self._lock = threading.Lock()
        # request_id -> live session, in start order (last = most recent).
        self._sessions: OrderedDict[str, _Live] = OrderedDict()
        # request_id -> latest DecodeTelemetry seen for that session.
        self._session_tel: dict[str, Any] = {}
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
            self._sessions[request_id] = _Live(request_id, mode, prompt_preview)

    def phase(self, request_id: str, phase: str) -> None:
        with self._lock:
            lv = self._sessions.get(request_id)
            if lv is not None:
                lv.phase = phase

    def tick(self, request_id: str, telemetry: Any | None = None, n: int = 1) -> None:
        """Called as tokens are emitted (``n`` tokens per call)."""
        with self._lock:
            lv = self._sessions.get(request_id)
            if lv is None:
                return
            lv.tokens += n
            lv.phase = "verify" if lv.mode == "diffusion" else "decode"
            if telemetry is not None:
                self._session_tel[request_id] = telemetry
            self._spark.append(round(lv.tok_s(), 1))

    def set_tokens(self, request_id: str, n: int) -> None:
        """Record a whole response's token count at once (non-streaming)."""
        with self._lock:
            lv = self._sessions.get(request_id)
            if lv is None:
                return
            lv.tokens = n
            self._spark.append(round(lv.tok_s(), 1))

    def finish(self, request_id: str, telemetry: Any | None = None,
               prompt_tokens: int = 0) -> None:
        with self._lock:
            lv = self._sessions.pop(request_id, None)
            if telemetry is None:
                telemetry = self._session_tel.pop(request_id, None)
            else:
                self._session_tel.pop(request_id, None)
            if lv is None:
                return
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
            self.accepted_from_draft += summary.get("draft_tokens_accepted", 0)
            self.pruned_positions += summary.get("pruned_draft_positions", 0)
            # Only Orthrus modes feed the AR-vs-diffusion speedup estimate.
            if lv.mode == "diffusion":
                self._diff_tps.append(tok_s)
            elif lv.mode == "ar":
                self._ar_tps.append(tok_s)
            app = summary.get("accepted_tokens_per_verification_pass",
                              1.0 if lv.mode != "proxy" else None)
            acc = summary.get("draft_acceptance_rate", 0.0)
            self._history.appendleft({
                "request_id": lv.request_id,
                "mode": lv.mode,
                "tokens": lv.tokens,
                "tok_s": round(tok_s, 1),
                "ms": round(elapsed * 1000),
                "accepted_per_pass": round(app, 2) if app is not None else None,
                "acceptance_rate": round(acc, 3),
                "prompt_preview": lv.prompt_preview,
                "prompt_tokens": prompt_tokens,
                "pruned": summary.get("pruned_draft_positions", 0),
                "ar_lane_steps": summary.get("ar_lane_steps", 0),
            })

    # -- dashboard read --------------------------------------------------- #
    def _baseline(self) -> float:
        if self._ar_tps:
            return sum(self._ar_tps) / len(self._ar_tps)
        return self.baseline_tok_s

    def _current(self) -> _Live | None:
        """The most recently started session that is still running."""
        if not self._sessions:
            return None
        return next(reversed(self._sessions.values()))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lv = self._current()
            tel = self._session_tel.get(lv.request_id) if lv is not None else None
            app = tel.accepted_tokens_per_verification_pass if tel is not None else 1.0
            acc = tel.draft_acceptance_rate if tel is not None else 0.0
            diff_tps = (sum(self._diff_tps) / len(self._diff_tps)) if self._diff_tps else 0.0
            base = self._baseline()
            speedup = (diff_tps / base) if (base and diff_tps) else None
            return {
                "live": {
                    "active": lv is not None,
                    "mode": lv.mode if lv else "",
                    "phase": lv.phase if lv else "idle",
                    "tokens": lv.tokens if lv else 0,
                    "tok_s": round(lv.tok_s(), 1) if lv else 0.0,
                    "accepted_per_pass": round(app, 2),
                    "acceptance_rate": round(acc, 3),
                    "prompt_preview": lv.prompt_preview if lv else "",
                    "concurrent": len(self._sessions),
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
