from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    elapsed_s: float = 0.0


@contextmanager
def elapsed_timer() -> Iterator[Timer]:
    timer = Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_s = time.perf_counter() - start
