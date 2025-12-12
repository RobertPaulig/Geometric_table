from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


TARGET_MAGIC = np.array([2, 8, 20, 28, 50, 82], dtype=float)


def cost_magic_l2(toy_magic: Sequence[int], target: np.ndarray | None = None, n_compare: int = 4) -> float:
    """
    Простейшая квадратичная метрика совпадения toy magic с эталонными.

    Используется в WS-тюнинге: сравниваются первые n_compare точек.
    """
    if target is None:
        target = TARGET_MAGIC
    if len(toy_magic) == 0:
        return float(1e9)
    n = min(n_compare, len(toy_magic), len(target))
    if n == 0:
        return float(1e9)
    tm = np.array(list(toy_magic)[:n], dtype=float)
    tg = target[:n]
    diff = tm - tg
    return float(np.sum(diff * diff))

