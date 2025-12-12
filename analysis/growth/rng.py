from __future__ import annotations

import hashlib
import numpy as np


BASE_SEED = 2025


def make_rng(label: str, offset: int = 0) -> np.random.Generator:
    """
    Детеминированный RNG для growth-стендов.

    label  — логическое имя (например, "cycle_stats", "temperature_scan").
    offset — опциональный сдвиг, если нужно несколько независимых RNG в одном скрипте.
    """
    h = hashlib.sha256(label.encode("utf-8")).digest()
    h_int = int.from_bytes(h[:8], "little")
    seed = (BASE_SEED + h_int + offset) & 0xFFFFFFFF
    return np.random.default_rng(seed)

