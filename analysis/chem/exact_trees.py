from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


def prufer_to_edges(code: Sequence[int], n: int) -> List[Edge]:
    """
    Convert a Prüfer code (length n-2) into a list of undirected edges.
    Vertices are labeled 0..n-1.
    """
    if n < 2:
        return []
    if len(code) != max(0, n - 2):
        raise ValueError(f"Expected Prüfer code length {n-2}, got {len(code)}")

    degree = [1] * n
    for x in code:
        if x < 0 or x >= n:
            raise ValueError(f"Invalid Prüfer label: {x}")
        degree[int(x)] += 1

    edges: List[Edge] = []
    code_list = list(int(x) for x in code)
    for x in code_list:
        leaf = next(i for i in range(n) if degree[i] == 1)
        edges.append((leaf, int(x)))
        degree[leaf] -= 1
        degree[int(x)] -= 1

    u = [i for i in range(n) if degree[i] == 1]
    edges.append((u[0], u[1]))
    return edges


def edges_to_adj(n: int, edges: Sequence[Edge]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i, j in edges:
        if i == j:
            continue
        a, b = (int(i), int(j))
        if a < 0 or b < 0 or a >= n or b >= n:
            raise ValueError("Edge out of bounds")
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def enumerate_labeled_trees(n: int) -> List[np.ndarray]:
    """
    Enumerate all labeled trees on n vertices via Prüfer sequences.
    Returns adjacency matrices (float 0/1).

    Counts:
      n=4 -> 16
      n=5 -> 125
    """
    if n < 1:
        return []
    if n == 1:
        return [np.zeros((1, 1), dtype=float)]
    if n == 2:
        adj = np.zeros((2, 2), dtype=float)
        adj[0, 1] = adj[1, 0] = 1.0
        return [adj]

    trees: List[np.ndarray] = []
    # iterate over all codes of length n-2 in base-n
    total = n ** (n - 2)
    for idx in range(total):
        x = idx
        code = []
        for _ in range(n - 2):
            code.append(int(x % n))
            x //= n
        edges = prufer_to_edges(code, n)
        trees.append(edges_to_adj(n, edges))
    return trees

