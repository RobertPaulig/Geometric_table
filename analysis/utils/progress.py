from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


@dataclass
class _FallbackProgress:
    total: int
    desc: str
    every: int = 100
    enabled: bool = True
    start: float = 0.0
    n: int = 0

    def __post_init__(self) -> None:
        self.start = time.perf_counter()

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        self.n += int(n)
        if self.n == 1 or self.n % self.every == 0 or self.n >= self.total:
            elapsed = time.perf_counter() - self.start
            rate = (self.n / elapsed) if elapsed > 0 else 0.0
            msg = f"[{self.desc}] {self.n}/{self.total} ({rate:.1f} it/s, {elapsed:.1f}s)"
            print(msg, file=sys.stderr, flush=True)

    def close(self) -> None:
        if not self.enabled:
            return
        self.update(0)


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    enabled: bool = True,
) -> Iterator[T]:
    """
    Wrap an iterable with a progress indicator.

    If tqdm is available, uses tqdm; otherwise prints periodic updates to stderr.
    """
    if not enabled:
        for x in iterable:
            yield x
        return

    try:
        from tqdm import tqdm  # type: ignore

        for x in tqdm(iterable, total=total, desc=desc, leave=True):
            yield x
        return
    except Exception:
        p = _FallbackProgress(total=int(total), desc=str(desc), every=100, enabled=True)
        try:
            for x in iterable:
                yield x
                p.update(1)
        finally:
            p.close()


def progress_update(pbar: Optional[object], n: int = 1) -> None:
    if pbar is None:
        return
    if hasattr(pbar, "update"):
        try:
            pbar.update(int(n))  # type: ignore[attr-defined]
        except Exception:
            return

