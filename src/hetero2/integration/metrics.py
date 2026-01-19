from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence


def _stable_float_token(x: float) -> str:
    v = float(x)
    if math.isfinite(v):
        return f"{v:.10g}"
    return "nan"


def curve_checksum_sha256(*, energy_grid: Sequence[float], values: Sequence[float]) -> str:
    h = hashlib.sha256()
    for e, v in zip(energy_grid, values, strict=False):
        h.update(_stable_float_token(e).encode("utf-8"))
        h.update(b",")
        h.update(_stable_float_token(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest().upper()


def percentile(values: Iterable[float], *, q: float) -> float:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return float("nan")
    if q <= 0:
        return float(vals[0])
    if q >= 100:
        return float(vals[-1])
    pos = (len(vals) - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

