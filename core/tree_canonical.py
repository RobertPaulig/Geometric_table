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


def _rooted_ahu_code(
    root: int, parent: int, adj_list: Sequence[Sequence[int]], labels: Dict[int, Tuple]
) -> Tuple:
    """
    AHU code for rooted trees as a nested tuple (much cheaper than string encoding).

    The empty tuple represents a leaf. Internal nodes are tuples of child codes, sorted.
    """
    children = [int(v) for v in adj_list[int(root)] if int(v) != int(parent)]
    encs = [_rooted_ahu_code(v, root, adj_list, labels) for v in children]
    encs.sort()
    code = tuple(encs)
    labels[int(root)] = code
    return code


def canonical_tree_permutation(adj_list: Sequence[Sequence[int]]) -> List[int]:
    """
    Canonical relabeling permutation for a connected tree using AHU encoding.

    Returns `perm` where `perm[new_index] = old_index`.
    """
    n = int(len(adj_list))
    if n <= 2:
        return list(range(n))

    adj_list_local = [list(nei) for nei in adj_list]
    centers = _tree_centers(adj_list_local)

    best_root = int(centers[0])
    best_code = None
    best_labels: Dict[int, Tuple] = {}
    for c in centers:
        labels: Dict[int, Tuple] = {}
        code = _rooted_ahu_code(int(c), -1, adj_list_local, labels)
        if best_code is None or code < best_code:
            best_code = code
            best_root = int(c)
            best_labels = labels

    perm: List[int] = []
    from collections import deque

    queue: deque[Tuple[int, int]] = deque([(best_root, -1)])
    while queue:
        u, parent = queue.popleft()
        perm.append(int(u))
        children = [int(v) for v in adj_list_local[int(u)] if int(v) != int(parent)]
        children.sort(key=lambda x: best_labels.get(int(x), ()))
        for v in children:
            queue.append((int(v), int(u)))

    return perm


def relabel_adj_list(adj_list: Sequence[Sequence[int]], perm: Sequence[int]) -> List[List[int]]:
    """
    Apply permutation `perm[new]=old` to an undirected adjacency list.

    Returns adjacency list in new labeling (neighbors sorted).
    """
    n = int(len(adj_list))
    if len(perm) != n:
        raise ValueError("perm length must match adj_list length")

    inv = [0] * n
    for new_i, old_i in enumerate(perm):
        inv[int(old_i)] = int(new_i)

    out: List[List[int]] = [[] for _ in range(n)]
    for new_u, old_u in enumerate(perm):
        out[new_u] = sorted(inv[int(v)] for v in adj_list[int(old_u)])
    return out


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
    perm = canonical_tree_permutation(adj_list)

    if len(perm) != n:
        raise ValueError("Tree relabeling failed (not connected tree?)")

    inv = {int(old): int(new) for new, old in enumerate(perm)}
    new_adj = np.zeros_like(adj, dtype=float)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                new_adj[inv[i], inv[j]] = 1.0
    return new_adj
