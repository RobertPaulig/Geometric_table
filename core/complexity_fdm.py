"""
complexity_fdm.py — FDM-подобный функционал сложности для деревьев.

Соавторы: R. Paulig + GPT-5.1 Thinking (OpenAI)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.tree_canonical import canonical_relabel_tree

# Default FDM parameters for tree complexity (v1 calibration):
# lambda controls depth penalty (0 < lambda < 1),
# q controls "fractal amplification" of larger subtrees.
LAMBDA_FDM_DEFAULT = 0.6
Q_FDM_DEFAULT = 1.5


def compute_fdm_complexity(
    adj_matrix: np.ndarray,
    lambda_weight: float | None = None,
    q: float | None = None,
    *,
    canonicalize_tree: bool = True,
) -> float:
    """
    FDM-подобная сложность графа, ориентированная на деревья.

    Идея:
    - рассматриваем дерево как фрактальную декомпозицию массы;
    - для каждой вершины v:
        * subtree_size[v] = размер поддерева (через DFS по остовному дереву);
        * mu_v = subtree_size[v] / n;
        * depth[v] = глубина в дереве.
    - функционал:
        C2(G) = [ sum_v (lambda^depth[v] * mu_v^q) ] * log2(1 + n),
      где lambda in (0, 1), q > 1.

    Для общего графа:
    - берём остовное дерево (BFS) и игнорируем циклы в FDM-части.
    """
    # Пустой граф или одиночный узел — почти нет фрактальной структуры
    if adj_matrix.size == 0:
        return 0.0

    n = adj_matrix.shape[0]
    if n <= 1:
        return 0.0

    if canonicalize_tree:
        # Make the complexity invariant to vertex labels for connected trees.
        # (For cycles/general graphs, we keep legacy behavior.)
        try:
            m = int(np.sum(adj_matrix) // 2)
            if m == n - 1:
                adj_matrix = canonical_relabel_tree(adj_matrix)
        except Exception:
            pass

    # 1. Строим остовное дерево (BFS) и родительский/глубинный массивы
    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    depth = np.zeros(n, dtype=int)

    # Берём 0-й узел как корень (при необходимости можно сделать параметром)
    root = 0
    visited[root] = True
    queue: deque[int] = deque([root])

    children: list[list[int]] = [[] for _ in range(n)]

    while queue:
        u = queue.popleft()
        neighbors = np.where(adj_matrix[u] > 0)[0]
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(v)
                queue.append(v)
        # Остальные рёбра (между уже посещёнными вершинами) считаем циклами и
        # игнорируем для построения дерева.

    # 2. DFS-порядок и размеры поддеревьев (bottom-up)
    subtree_size = np.ones(n, dtype=int)

    order: list[int] = []

    def dfs_build_order(u: int) -> None:
        order.append(u)
        for v in children[u]:
            dfs_build_order(v)

    dfs_build_order(root)

    # Идём в обратном порядке: листья -> корень
    for u in reversed(order):
        total = 1
        for v in children[u]:
            total += subtree_size[v]
        subtree_size[u] = total

    # 3. Нормированная масса и FDM-функционал
    n_total = float(subtree_size[root])
    if n_total <= 0.0:
        return 0.0

    mu = subtree_size.astype(float) / n_total

    # R&D-параметры FDM-слоя (связаны с Fgeom-теорией):
    # - lambda_weight — глубинный штраф (чем больше глубина, тем меньше вклад вершины);
    # - q — фрактальное усиление по размеру поддерева.
    if lambda_weight is None:
        lambda_weight = LAMBDA_FDM_DEFAULT
    if q is None:
        q = Q_FDM_DEFAULT  # "фрактальный индекс"

    c2_raw = 0.0
    for u in range(n):
        mu_u = mu[u]
        if mu_u <= 0.0:
            continue
        c2_raw += (lambda_weight ** depth[u]) * (mu_u ** q)

    c2 = c2_raw * math.log2(1.0 + n_total)
    return float(c2)
