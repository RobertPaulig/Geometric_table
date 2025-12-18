from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, MutableMapping


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_sec(x: float) -> str:
    x = float(x)
    if x < 0:
        x = 0.0
    if x < 60:
        return f"{x:.3f}s"
    m = int(x // 60)
    s = x - 60 * m
    if m < 60:
        return f"{m:d}m {s:.1f}s"
    h = int(m // 60)
    mm = int(m - 60 * h)
    return f"{h:d}h {mm:d}m {s:.0f}s"


@contextmanager
def timed(name: str, acc: MutableMapping[str, float]) -> Iterator[float]:
    t0 = time.perf_counter()
    try:
        yield 0.0
    finally:
        dt = time.perf_counter() - t0
        acc[name] = float(acc.get(name, 0.0)) + float(dt)

