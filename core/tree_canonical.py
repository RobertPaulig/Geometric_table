from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _adj_list_from_adj(adj: np.ndarray) -> List[List[int]]:
    n = int(adj.shape[0])
    out: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        nbrs = np.where(adj[i] > 0)[0]
        out[i] = [int(x) for x in nbrs.tolist()]
    return out


def _tree_centers(adj_list: List[List[int]]) -> List[int]:
    n = len(adj_list)
    deg = [len(adj_list[i]) for i in range(n)]
    leaves = [i for i, d in enumerate(deg) if d <= 1]
    removed = len(leaves)
    while removed < n:
        new_leaves: List[int] = []
        for u in leaves:
            deg[u] = 0
            for v in adj_list[u]:
                if deg[v] > 0:
                    deg[v] -= 1
                    if deg[v] == 1:
                        new_leaves.append(v)
        removed += len(new_leaves)
        leaves = new_leaves
    return leaves if leaves else [0]


def _rooted_ahu_encoding(
    root: int, parent: int, adj_list: List[List[int]], labels: Dict[int, str]
) -> str:
    children = [v for v in adj_list[root] if v != parent]
    encs = [_rooted_ahu_encoding(v, root, adj_list, labels) for v in children]
    encs.sort()
    code = "(" + "".join(encs) + ")"
    labels[root] = code
    return code


def canonical_relabel_tree(adj: np.ndarray) -> np.ndarray:
    """
    Canonical relabeling for a connected tree using AHU encoding.

    Returns a permuted adjacency matrix that is invariant to input labeling.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    n = int(adj.shape[0])
    if n <= 2:
        return np.asarray(adj, dtype=float)

    adj_list = _adj_list_from_adj(adj)
    centers = _tree_centers(adj_list)

    # Choose canonical center by minimal AHU code.
    best_root = centers[0]
    best_code = None
    best_labels: Dict[int, str] = {}
    for c in centers:
        labels: Dict[int, str] = {}
        code = _rooted_ahu_encoding(c, -1, adj_list, labels)
        if best_code is None or code < best_code:
            best_code = code
            best_root = c
            best_labels = labels

    # Deterministic BFS order from best_root, ordering children by AHU labels.
    perm: List[int] = []
    queue: List[Tuple[int, int]] = [(best_root, -1)]
    while queue:
        u, parent = queue.pop(0)
        perm.append(u)
        children = [v for v in adj_list[u] if v != parent]
        children.sort(key=lambda x: best_labels.get(x, ""))
        for v in children:
            queue.append((v, u))

    if len(perm) != n:
        raise ValueError("Tree relabeling failed (not connected tree?)")

    inv = {old: new for new, old in enumerate(perm)}
    new_adj = np.zeros_like(adj, dtype=float)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                new_adj[inv[i], inv[j]] = 1.0
    return new_adj

