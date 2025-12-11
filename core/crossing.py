from __future__ import annotations

"""
crossing.py — CY-1/step4: toy crossing-number для малых графов.

Модель: вершины лежат на окружности, рёбра — хорды; пересечения считаются
по переплетению интервалов. Для n <= max_exact_n используется полный
перебор перестановок (точный crossing-number в этой модели), для больших
графов возможен стохастический режим.
"""

from itertools import permutations
from typing import Sequence, Tuple, Dict, List, Optional

import numpy as np

Edge = Tuple[int, int]


def _extract_edges(adj: np.ndarray) -> List[Edge]:
    n = adj.shape[0]
    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append((i, j))
    return edges


def _count_crossings_for_order(edges: Sequence[Edge], order: Sequence[int]) -> int:
    """
    Число пересечений для данного порядка вершин на окружности.

    Вершины лежат на окружности в порядке `order`, рёбра — хорды.
    """
    pos: Dict[int, int] = {v: idx for idx, v in enumerate(order)}

    def edge_interval(e: Edge) -> Tuple[int, int]:
        a, b = e
        ia, ib = pos[a], pos[b]
        return (ia, ib) if ia < ib else (ib, ia)

    intervals = [edge_interval(e) for e in edges]
    m = len(edges)
    crossings = 0

    for i in range(m):
        a1, b1 = intervals[i]
        for j in range(i + 1, m):
            a2, b2 = intervals[j]
            # Взаимное переплетение интервалов => пересечение хорд
            if (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1):
                crossings += 1
    return crossings


def estimate_crossing_number_circle(
    adj: np.ndarray,
    max_exact_n: int = 8,
    n_random: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, bool]:
    """
    Оценка (или точное значение для малых n) crossing-number
    в модели «вершины на окружности, рёбра — хорды».

    Возвращает:
      crossing_min: минимальное найденное число пересечений
      is_exact: True, если был перебор всех перестановок (точное значение),
                False, если только случайный поиск.
    """
    n = adj.shape[0]
    if n <= 1:
        return 0, True

    edges = _extract_edges(adj)
    if len(edges) <= 1:
        return 0, True

    vertices = list(range(n))

    # Точный перебор для малых n
    if n <= max_exact_n:
        crossing_min: Optional[int] = None
        for perm in permutations(vertices):
            c = _count_crossings_for_order(edges, perm)
            if crossing_min is None or c < crossing_min:
                crossing_min = c
                if crossing_min == 0:
                    break
        return int(crossing_min or 0), True

    # Стохастическая оценка для более крупных графов
    if rng is None:
        rng = np.random.default_rng()

    crossing_min: Optional[int] = None
    for _ in range(n_random):
        perm = list(vertices)
        rng.shuffle(perm)
        c = _count_crossings_for_order(edges, perm)
        if crossing_min is None or c < crossing_min:
            crossing_min = c
            if crossing_min == 0:
                break

    return int(crossing_min or 0), False

