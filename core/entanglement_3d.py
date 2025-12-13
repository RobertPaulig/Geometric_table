from __future__ import annotations

from typing import List, Tuple

import numpy as np


def segment_segment_distance(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Минимальная дистанция между двумя отрезками в 3D.
    Основано на проекциях на линию и клиппинге параметров в [0,1].
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w0))
    e = float(np.dot(v, w0))

    denom = a * c - b * b
    if denom <= 1e-12:
        # Почти параллельные отрезки
        t = 0.0
        s = max(0.0, min(1.0, -e / max(c, 1e-12)))
    else:
        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom
        t = max(0.0, min(1.0, t))
        s = max(0.0, min(1.0, s))

    pc = p1 + t * u
    qc = q1 + s * v
    return float(np.linalg.norm(pc - qc))


def entanglement_score(
    pos: np.ndarray,
    edges: List[Tuple[int, int]],
    *,
    sigma: float = 0.25,
    ignore_adjacent: bool = True,
) -> float:
    """
    Простая 3D-метрика "запутанности": суммируем exp(-(d/sigma)^2)
    по парам рёбер (не смежным, если ignore_adjacent=True).
    """
    pts = np.asarray(pos, dtype=float)
    n = pts.shape[0]
    if n <= 1 or len(edges) <= 1:
        return 0.0

    m = len(edges)
    score = 0.0
    inv_sigma2 = 1.0 / (sigma * sigma + 1e-12)

    for i in range(m):
        a1, b1 = edges[i]
        for j in range(i + 1, m):
            a2, b2 = edges[j]
            if ignore_adjacent and (
                a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2
            ):
                continue
            d = segment_segment_distance(pts[a1], pts[b1], pts[a2], pts[b2])
            val = float(np.exp(-d * d * inv_sigma2))
            score += val
    return float(score)

